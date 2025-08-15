import argparse
import logging
import os
import json
import time
import typing
import uuid
import timesfm

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.periodicsequence import PeriodicImpulse
import pandas as pd
import numpy as np
from apache_beam.utils.timestamp import Timestamp
from apache_beam.ml.inference.base import RunInference
from google.cloud import storage
from apache_beam.ml.ts.finetuning_torch import FinetuningConfig
from apache_beam.ml.ts.inference_model_handler import DynamicTimesFmModelHandler, LatestModelCheckpointLoader
from apache_beam.ml.ts.finetuning_components import BatchContinuousAndOrderedFn, RunFinetuningFn
from apache_beam.ml.ts.llm_classifier import LLMClassifierFn
from apache_beam.examples.inference.anomaly_detection.timesfm_anomaly_detection.ordered_sliding_window import OrderedSlidingWindowFn, FillGapsFn
from apache_beam.ml.ts.finetuning_torch import FinetuningConfig
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.utils import WatchFilePattern
from apache_beam.transforms.window import TimestampedValue

from apache_beam.transforms.window import GlobalWindows
from apache_beam.transforms.trigger import AccumulationMode, AfterCount, Repeatedly
from apache_beam.transforms.window import FixedWindows

# =================================================================
# 1. SCRIPT CONFIGURATION
#    (No command-line arguments needed)
# =================================================================
GCP_PROJECT = "apache-beam-testing" # ⬅️ SET YOUR GCP PROJECT ID
GCP_REGION = "us-central1"           # ⬅️ SET YOUR GCP REGION
INPUT_BUCKET = "ashok-testing"       # ⬅️ SET YOUR INPUT BUCKET
INPUT_FOLDER = "times-series-datasets"            # ⬅️ SET FOLDER IN BUCKET (or "" for root)
OUTPUT_BUCKET = "ashok-testing"      # ⬅️ SET YOUR OUTPUT BUCKET
TEMP_LOCATION = f"gs://{OUTPUT_BUCKET}/temp"
STAGING_LOCATION = f"gs://{OUTPUT_BUCKET}/staging"
# This is the base directory where folders for each fine-tuned model will be created.
FINETUNED_MODEL_BASE_PREFIX = "finetuned-models"
# IMPORTANT: Provide the path to the setup.py file for your project.
# Dataflow needs this to install dependencies like 'timesfm' on the workers.
SETUP_FILE_PATH = "/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/setup.py" # ⬅️ SET ABSOLUTE PATH TO YOUR setup.py

# =================================================================
class CustomJsonEncoder(json.JSONEncoder):
    """A custom JSON encoder that knows how to handle Beam's Timestamp objects."""
    def default(self, obj):
        if isinstance(obj, Timestamp):
            # Convert Timestamp to a standard, readable ISO 8601 string format
            return obj.micros // 1e6
        # For all other types, fall back to the default behavior
        if isinstance(obj, np.integer):
            return int(obj)
        # 3. Handle NumPy float types (this will fix your float32 error)
        if isinstance(obj, np.floating):
            return float(obj)
        # 4. Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # For all other types, fall back to the default behavior
        return super().default(obj)

class WritePlotDataToGCS(beam.DoFn):
    """
    Writes plotting data to a GCS file as a side effect and passes
    the original element downstream. This is safe for Dataflow.
    """
    def __init__(self, output_path):
        self._output_path = output_path
        self._file_handle = None

    def setup(self):
        # Use Beam's FileSystems API to open a writeable file on GCS
        self._file_handle = FileSystems.create(self._output_path)

    def process(self, element):
        _original_window, payload_dict = element
        json_record = json.dumps(payload_dict, cls=CustomJsonEncoder) + '\n'
        # Write to the GCS file handle (requires bytes)
        self._file_handle.write(json_record.encode('utf-8'))
        # Pass the original element through unmodified
        yield element

    def teardown(self):
        # The FileSystems API handles closing the file
        if self._file_handle:
            self._file_handle.close()

def print_and_pass_through(label):
    def logger(element):
        logging.info(f"--- {label} --- \nELEMENT: %s", element)
        return element
    return logger

def format_for_llm(result_tuple):
    """
    Takes the output of RunInference (a PredictionResult) and formats it
    into the dictionary structure needed by the LLMClassifierFn.
    """
    # prediction_result is a named tuple: (example, inference)
    original_window_data, list_of_anomalies = result_tuple

    key, (window_start_ts, _, values_array) = original_window_data

    return (key, {
        'key': key,
        'window_start_ts': window_start_ts,
        'values_array': values_array,
        'anomalies': list_of_anomalies if list_of_anomalies else []
    })

class WriteSingleItemToGCS(beam.DoFn):
    """A DoFn to write a single item from a PCollection to a GCS file."""
    def __init__(self, output_path):
        self._output_path = output_path

    def process(self, element):
        # element is the single model path string
        logging.info(f"Writing final model path to {self._output_path}: {element}")
        with FileSystems.create(self._output_path) as f:
            f.write(str(element).encode('utf-8'))
        # We don't need to yield anything downstream.

## =================================================================
## 3. PIPELINE DEFINITIONS (Modified for Hyperparameter Tuning)
## =================================================================

def create_finetuning_pipeline(p,
                               input_data,
                               model_path_output_path,
                               dedicated_model_prefix,
                               horizon_len,
                               num_epochs,
                               context_len=512):
    """
    Defines the finetuning pipeline.
    This pipeline takes data, runs inference with the initial model to get
    anomaly scores, uses an LLM to classify them, and then fine-tunes the
    model on the classified data. It does NOT save plot data for the
    initial model's inference run.
    """
    WINDOW_SIZE = context_len + horizon_len
    SLIDE_INTERVAL = horizon_len
    EXPECTED_INTERVAL = 1
    FINETUNING_BATCH_SIZE = int(len(input_data) * 0.8)
    INITIAL_MODEL = "google/timesfm-1.0-200m-pytorch"
    API_KEY = os.getenv("GEMINI_API_KEY")

    # Configuration for the fine-tuning process
    FINETUNE_CONFIG = FinetuningConfig(
        batch_size=128,
        num_epochs=num_epochs,  # Use the provided number of epochs
        learning_rate=1e-4,
        use_wandb=False,
        freq_type=0,
        log_every_n_steps=10,
        val_check_interval=0.5,
        use_quantile_loss=True)

    # Model hyperparameters, including the specified horizon length
    hparams = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=horizon_len,
        context_len=context_len)

    # --- Pipeline Graph ---
    windowed_data = (
        p
        | 'Impulse' >> PeriodicImpulse(data=input_data, fire_interval=0.001)
        | 'AddKey' >> beam.WithKeys(lambda x: 0)
        | 'SlidingWindow' >> beam.ParDo(
            OrderedSlidingWindowFn(WINDOW_SIZE, SLIDE_INTERVAL))
        | 'FillGaps' >> beam.ParDo(FillGapsFn(EXPECTED_INTERVAL)))

    # Run inference on the initial model to get data for the LLM classifier
    inference_results = (
        windowed_data
        | "DetectAnomalies" >> RunInference(
            DynamicTimesFmModelHandler(model_uri=INITIAL_MODEL, hparams=hparams)))

    # Format the results for the LLM
    data_for_llm = (
        inference_results
        | "FormatForLLM" >> beam.Map(format_for_llm))

    # Use the LLM to classify anomalies and prepare data for fine-tuning
    finetuning_input = (
        data_for_llm
        | "LLMClassifier" >> beam.ParDo(
            LLMClassifierFn(
                secret=API_KEY,
                slide_interval=SLIDE_INTERVAL,
                expected_interval_secs=EXPECTED_INTERVAL))
        | "KeyForBatching" >> beam.WithKeys(lambda _: "finetune_batch")
        | "BatchForFinetuning" >> beam.ParDo(
            BatchContinuousAndOrderedFn(
                FINETUNING_BATCH_SIZE, expected_interval_seconds=1)))

    # Run the fine-tuning job
    finetuned_model_path = (
        finetuning_input
        | "RunFinetuning" >> beam.ParDo(
            RunFinetuningFn(
                initial_model_path=INITIAL_MODEL,
                finetuned_model_bucket=OUTPUT_BUCKET,
                finetuned_model_prefix=dedicated_model_prefix,
                hparams=hparams,
                config=FINETUNE_CONFIG,
                noise_level=0.01)))

    # Save the path of the newly fine-tuned model to a GCS file
    _ = (
        finetuned_model_path
        | "WriteFinetunedModelPath" >> beam.ParDo(
            WriteSingleItemToGCS(model_path_output_path)))


def create_inference_pipeline(p,
                              input_data,
                              plot_output_path,
                              finetuned_model_path,
                              horizon_len,
                              context_len=512):
    """
    Defines the inference pipeline.
    This pipeline uses a fine-tuned model to run inference on the full dataset
    and saves the results as plotting data.
    """
    WINDOW_SIZE = context_len + horizon_len
    SLIDE_INTERVAL = horizon_len
    EXPECTED_INTERVAL = 1

    # Model hyperparameters must match the fine-tuned model
    hparams = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=horizon_len,
        context_len=context_len)

    # --- Pipeline Graph ---
    windowed_data = (
        p
        | 'Impulse_FT' >> PeriodicImpulse(data=input_data, fire_interval=0.001)
        | "AddKey_FT" >> beam.WithKeys(lambda x: 0)
        | "SlidingWindow_FT" >> beam.ParDo(
            OrderedSlidingWindowFn(WINDOW_SIZE, SLIDE_INTERVAL))
        | "FillGaps_FT" >> beam.ParDo(FillGapsFn(EXPECTED_INTERVAL)))

    # Run inference with the fine-tuned model
    inference_results = (
        windowed_data
        | "DetectAnomalies_FT" >> RunInference(
            DynamicTimesFmModelHandler(
                model_uri=finetuned_model_path, hparams=hparams)))

    # Save the results for plotting
    _ = (
        inference_results
        | "WriteFinetunedPlotData" >> beam.ParDo(
            WritePlotDataToGCS(plot_output_path)))


## =================================================================
## 4. MAIN ORCHESTRATION SCRIPT
## =================================================================
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # --- Base Configuration ---
    pipeline_options = {
        "runner": "DataflowRunner",
        "project": GCP_PROJECT,
        "region": GCP_REGION,
        "temp_location": TEMP_LOCATION,
        "staging_location": STAGING_LOCATION,
        "setup_file": SETUP_FILE_PATH,
        "sdk_location": "dist/apache_beam-2.67.0.dev0.tar.gz",
        "streaming": True,
        "machine_type" : "g2-standard-4",
        "dataflow_service_options" : "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver"
    } 

    # --- Load Data ---
    file_to_process = "WSD_94.csv"
    file_uri = f"gs://{INPUT_BUCKET}/{INPUT_FOLDER}/{file_to_process}"
    base_filename = os.path.basename(file_uri).replace('.csv', '')

    try:
        df = pd.read_csv(file_uri)
        data_values = df['Data'].to_numpy()
        input_data = [(Timestamp(i), value) for i, value in enumerate(data_values)]
        logging.info(f"✅ Successfully loaded {len(input_data)} records from {file_uri}")
    except Exception as e:
        logging.error(f"❌ Failed to load or parse {file_uri}. Aborting. Error: {e}")
        exit()

    # --- STAGE 1: Launch all finetuning jobs in parallel ---
    experiments_metadata = []
    horizons_to_test = [] # [16, 32, 64, 96]
    epochs_to_test = [1, 4]

    # Create a list of all experiment configurations
    all_experiments = []
    for horizon in horizons_to_test:
        all_experiments.append({
            "experiment_type": "horizon", "param_value": horizon,
            "horizon_len": horizon, "num_epochs": 1
        })
    for epochs in epochs_to_test:
        all_experiments.append({
            "experiment_type": "epoch", "param_value": epochs,
            "horizon_len": 128, "num_epochs": epochs
        })

    logging.info(f"\n{'#'*80}\n# LAUNCHING ALL {len(all_experiments)} FINETUNING JOBS IN PARALLEL\n{'#'*80}")

    for exp_config in all_experiments:
        exp_type = exp_config["experiment_type"]
        param_val = exp_config["param_value"]

        job_options = pipeline_options.copy()
        job_options["job_name"] = (
            f"finetune-{exp_type}-{param_val}-"
            f"{base_filename.replace('_', '-')}-{str(uuid.uuid4())[:4]}").lower()

        experiment_folder = f"{exp_type}_{param_val}"
        dedicated_model_prefix = os.path.join(FINETUNED_MODEL_BASE_PREFIX, base_filename, experiment_folder)
        model_path_output_file = os.path.join(TEMP_LOCATION, "model_paths", f"{job_options['job_name']}.txt")

        logging.info(f"  - Submitting finetuning job: {job_options['job_name']}")
        p = beam.Pipeline(options=PipelineOptions.from_dictionary(job_options))
        create_finetuning_pipeline(
            p=p, input_data=input_data,
            model_path_output_path=model_path_output_file,
            dedicated_model_prefix=dedicated_model_prefix,
            horizon_len=exp_config["horizon_len"],
            num_epochs=exp_config["num_epochs"])
        result = p.run()

        experiments_metadata.append({
            "job_result": result,
            "job_name": job_options["job_name"],
            "model_path_output_file": model_path_output_file,
            "experiment_folder": experiment_folder,
            "horizon_len": exp_config["horizon_len"],
            **exp_config
        })
        logging.info(f"  - Job {job_options['job_name']} is running (ID: {result.job_id()}).")

    # --- STAGE 2: Wait for all finetuning jobs to complete ---
    logging.info(f"\n{'#'*80}\n# WAITING FOR ALL FINETUNING JOBS TO COMPLETE\n{'#'*80}")
    for item in experiments_metadata:
        item['job_result'].wait_until_finish(duration=604800) # 1 week in seconds
        logging.info(f"  - Finetuning job {item['job_name']} finished with state: {item['job_result'].state}")

    # --- STAGE 3: Launch all inference jobs in parallel ---
    inference_jobs = []
    successful_finetunes = []

    logging.info(f"\n{'#'*80}\n# PREPARING TO LAUNCH INFERENCE JOBS\n{'#'*80}")
    for item in experiments_metadata:
        if item['job_result'].state == beam.runners.runner.PipelineState.DONE:
            try:
                with FileSystems.open(item['model_path_output_file']) as f:
                    item['finetuned_model_gcs_path'] = f.read().decode('utf-8').strip()
                logging.info(f"  - Read model path for {item['job_name']}: {item['finetuned_model_gcs_path']}")
                successful_finetunes.append(item)
            except Exception as e:
                logging.error(f"  - Failed to read model path for {item['job_name']}. Skipping. Error: {e}")
        else:
            logging.error(f"  - Finetuning job {item['job_name']} FAILED. Skipping inference.")

    for item in successful_finetunes:
        job_options = pipeline_options.copy()
        job_options["job_name"] = (
            f"inference-{item['experiment_type']}-{item['param_value']}-"
            f"{base_filename.replace('_', '-')}-{str(uuid.uuid4())[:4]}").lower()

        finetuned_plot_path = os.path.join(
            f"gs://{OUTPUT_BUCKET}", "plot_data", base_filename,
            item['experiment_folder'], f"{base_filename}_finetuned.jsonl")

        logging.info(f"  - Submitting inference job: {job_options['job_name']}")
        p = beam.Pipeline(options=PipelineOptions.from_dictionary(job_options))
        create_inference_pipeline(
            p=p, input_data=input_data,
            plot_output_path=finetuned_plot_path,
            finetuned_model_path=item['finetuned_model_gcs_path'],
            horizon_len=item['horizon_len'])
        result = p.run()
        inference_jobs.append(result)
        logging.info(f"  - Job {job_options['job_name']} is running (ID: {result.job_id()}).")

    # --- STAGE 4: Wait for all inference jobs to complete ---
    if inference_jobs:
        logging.info(f"\n{'#'*80}\n# WAITING FOR ALL INFERENCE JOBS TO COMPLETE\n{'#'*80}")
        for job_result in inference_jobs:
            job_result.wait_until_finish(duration=604800) # 1 week in seconds
            logging.info(f"  - Inference job {job_result.job_id()} finished with state: {job_result.state}")

    logging.info(f"\n{'#'*80}\n# ALL HYPERPARAMETER EXPERIMENTS COMPLETED\n{'#'*80}")
