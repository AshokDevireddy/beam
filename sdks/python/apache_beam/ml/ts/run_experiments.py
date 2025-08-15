import argparse
import logging
import os
import re
import random
import string
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
def sanitize_job_name_part(name_part):
    """
    Cleans a string to be a valid part of a Dataflow job name.
    It must only contain [-a-z0-9].
    """
    # Convert to lowercase first.
    name_part = name_part.lower()
    # Replace any character that is not a lowercase letter, number, or hyphen
    # with a hyphen.
    sanitized = re.sub(r'[^a-z0-9-]+', '-', name_part)
    # Remove leading or trailing hyphens.
    sanitized = sanitized.strip('-')
    # If the name is empty after sanitization (e.g., "."), generate a random one.
    if not sanitized:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    # Truncate to avoid exceeding the 63-character job name limit.
    return sanitized[:40]


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
## 3. PIPELINE DEFINITIONS (Using Your Structure)
## =================================================================


def create_finetuning_pipeline(p, input_data, plot_output_path, model_path_output_path, dedicated_model_prefix):
    """Defines the first pipeline using your exact PeriodicImpulse structure."""
    CONTEXT_LEN, HORIZON_LEN = 512, 128
    WINDOW_SIZE, SLIDE_INTERVAL, EXPECTED_INTERVAL = CONTEXT_LEN + HORIZON_LEN, HORIZON_LEN, 1
    FINETUNING_BATCH_SIZE = int(len(input_data) * 0.8)
    INITIAL_MODEL = "google/timesfm-1.0-200m-pytorch"
    API_KEY = os.getenv("GEMINI_API_KEY") # Replace with your key or secret management

    FINETUNE_CONFIG = FinetuningConfig(
        batch_size=128, num_epochs=1, learning_rate=1e-4, use_wandb=False,
        freq_type=0, log_every_n_steps=10, val_check_interval=0.5, use_quantile_loss=True
    )
    # Your hparams definition might be missing `timesfm` import
    try:
        import timesfm
        hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32, horizon_len=HORIZON_LEN, context_len=CONTEXT_LEN)
    except ImportError:
        hparams = {}
    
    # --- Your exact pipeline graph ---
    windowed_data = (
        p
        | PeriodicImpulse(
            data=input_data, 
            fire_interval=0.001)
        | "AddKey" >> beam.WithKeys(lambda x: 0)
        | "ApplySlidingWindow" >> beam.ParDo(OrderedSlidingWindowFn(WINDOW_SIZE, SLIDE_INTERVAL))
        | "FillGaps" >> beam.ParDo(FillGapsFn(EXPECTED_INTERVAL))
        | "PrintWindowedData" >> beam.Map(print_and_pass_through("Windowed Data"))
    )

    inference_results = (
        windowed_data
        | "DetectAnomalies" >> RunInference(DynamicTimesFmModelHandler(model_uri=INITIAL_MODEL, hparams=hparams))
        | "PrintInference" >> beam.Map(print_and_pass_through("Inference Results"))
    )

    # Branched output for plotting
    _ = (
        inference_results
        | "WritePlotData" >> beam.ParDo(WritePlotDataToGCS(plot_output_path))
    )

    # Main branch continues for finetuning
    data_for_llm = (
        inference_results
        | "FormatForLLM" >> beam.Map(format_for_llm)
        | "PrintDataforLLM" >> beam.Map(print_and_pass_through("Data for LLM"))
    )

    finetuning_input = (
        data_for_llm
        | "LLMClassifier" >> beam.ParDo(
            LLMClassifierFn(
                secret=API_KEY, 
                slide_interval=SLIDE_INTERVAL,
                expected_interval_secs=EXPECTED_INTERVAL
                )
            )
        | "KeyForBatching" >> beam.WithKeys(lambda _: "finetune_batch")
        | "BatchForFinetuning" >> beam.ParDo(
            BatchContinuousAndOrderedFn(
                FINETUNING_BATCH_SIZE, 
                expected_interval_seconds=1
                )
            )
        | "PrintFinetuningJobInput" >> beam.Map(print_and_pass_through("Finetuning Job Input"))

    )

    finetuned_model_path = (
        finetuning_input
        | "RunFinetuning" >> beam.ParDo(
            RunFinetuningFn(
                initial_model_path=INITIAL_MODEL,
                finetuned_model_bucket=OUTPUT_BUCKET,
                finetuned_model_prefix=dedicated_model_prefix,
                hparams=hparams, config=FINETUNE_CONFIG
            )
        )
    )
    _ = (
        finetuned_model_path
        | "WriteFinetunedModelPath" >> beam.ParDo(
            WriteSingleItemToGCS(model_path_output_path)
        )
    )

def create_inference_pipeline(p, input_data, plot_output_path, finetuned_model_path):
    """Defines the second pipeline for inference with the new model."""
    CONTEXT_LEN, HORIZON_LEN = 512, 128
    WINDOW_SIZE, SLIDE_INTERVAL, EXPECTED_INTERVAL = CONTEXT_LEN + HORIZON_LEN, HORIZON_LEN, 1
    try:
        import timesfm
        hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32, horizon_len=HORIZON_LEN, context_len=CONTEXT_LEN)
    except ImportError:
        hparams = {}

    windowed_data = (
        p
        | PeriodicImpulse(
            data=input_data, 
            fire_interval=0.001)
        | "AddKey_FT" >> beam.WithKeys(lambda x: 0)
        | "ApplySlidingWindow_FT" >> beam.ParDo(OrderedSlidingWindowFn(WINDOW_SIZE, SLIDE_INTERVAL))
        | "FillGaps_FT" >> beam.ParDo(FillGapsFn(EXPECTED_INTERVAL))
    )

    inference_results = (
        windowed_data
        | "DetectAnomalies_FT" >> RunInference(DynamicTimesFmModelHandler(model_uri=finetuned_model_path, hparams=hparams))
    )

    _ = (
        inference_results
        | "WriteFinetunedPlotData" >> beam.ParDo(WritePlotDataToGCS(plot_output_path))
    )


## =================================================================
## 4. MAIN ORCHESTRATION SCRIPT
## =================================================================
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

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
        "disk_size_gb": 25,
        "dataflow_service_options" : "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver"
    }

    storage_client = storage.Client(project=GCP_PROJECT)
    blobs = storage_client.list_blobs(INPUT_BUCKET, prefix=INPUT_FOLDER)
    csv_files = [blob for blob in blobs if blob.name.endswith('.csv') and not blob.name.endswith('/')]

    if not csv_files:
        logging.error(f"No CSV files found in gs://{INPUT_BUCKET}/{INPUT_FOLDER}")
    else:
        logging.info(f"✅ Found {len(csv_files)} CSV files to process in parallel.")

    # --- STAGE 1: Launch all finetuning jobs in parallel ---
    finetuning_jobs = []
    dataset_metadata = []

    for blob in csv_files:
        file_uri = f"gs://{blob.bucket.name}/{blob.name}"
        base_filename = os.path.basename(blob.name).replace('.csv', '')
        logging.info(f"\n{'='*60}\n🚀 Preparing finetuning job for: {base_filename}\n{'='*60}")

        try:
            df = pd.read_csv(file_uri)
            data_values = df['Data'].to_numpy()
            input_data = [(Timestamp(i), value) for i, value in enumerate(data_values)]
            input_data = input_data[:8000]  # Limit to first 2000 points for testing
            logging.info(f"  - Successfully loaded {len(input_data)} records.")
        except Exception as e:
            logging.error(f"  - Failed to load or parse {file_uri}. Skipping. Error: {e}")
            continue

        job_options = pipeline_options.copy()
        sanitized_filename = sanitize_job_name_part(base_filename)
        job_options["job_name"] = f"finetune-{sanitized_filename}-{str(uuid.uuid4())[:4]}"

        original_plot_path = os.path.join(f"gs://{OUTPUT_BUCKET}", "plot_data", f"{base_filename}_original.jsonl")
        dedicated_model_prefix = os.path.join(FINETUNED_MODEL_BASE_PREFIX, base_filename)
        model_path_output_file = os.path.join(TEMP_LOCATION, "model_paths", f"{job_options['job_name']}.txt")

        logging.info(f"  - Submitting finetuning job: {job_options['job_name']}")
        p = beam.Pipeline(options=PipelineOptions.from_dictionary(job_options))
        create_finetuning_pipeline(p, input_data, original_plot_path, model_path_output_file, dedicated_model_prefix)
        result = p.run()
        finetuning_jobs.append(result)
        dataset_metadata.append({
            "base_filename": base_filename,
            "input_data": input_data,
            "model_path_output_file": model_path_output_file,
            "job_result": result
        })
        logging.info(f"  - Job {job_options['job_name']} is running (ID: {result.job_id()}).")

    # --- STAGE 2: Wait for all finetuning jobs to complete ---
    if finetuning_jobs:
        logging.info(f"\n{'='*80}\nWaiting for all {len(finetuning_jobs)} finetuning jobs to complete...\n{'='*80}")
        for item in dataset_metadata:
            job_result = item['job_result']
            try:
                job_result.wait_until_finish()
                logging.info(f"  - ✅ Finetuning job for {item['base_filename']} finished with state: {job_result.state}")
            except Exception as e:
                logging.error(f"  - ❌ Finetuning job for {item['base_filename']} FAILED.")
                logging.error(f"    - Error: {e}")
                # We can inspect the job state even if it failed.
                logging.error(f"    - Final job state: {job_result.state}")

    # --- STAGE 3: Launch all inference jobs in parallel ---
    inference_jobs = []
    successful_finetunes = []

    logging.info(f"\n{'='*80}\nPreparing to launch inference jobs...\n{'='*80}")
    for item in dataset_metadata:
        if item['job_result'].state == beam.runners.runner.PipelineState.DONE:
            try:
                with FileSystems.open(item['model_path_output_file']) as f:
                    finetuned_model_gcs_path = f.read().decode('utf-8').strip()
                logging.info(f"  - Successfully read model path for {item['base_filename']}: {finetuned_model_gcs_path}")
                item['finetuned_model_gcs_path'] = finetuned_model_gcs_path
                successful_finetunes.append(item)
            except Exception as e:
                logging.error(f"  - Failed to read model path for {item['base_filename']}. Skipping inference. Error: {e}")
        else:
            logging.error(f"  - Finetuning job for {item['base_filename']} FAILED. Skipping inference.")

    for item in successful_finetunes:
        base_filename = item['base_filename']
        job_options = pipeline_options.copy()
        sanitized_filename = sanitize_job_name_part(base_filename)
        job_options["job_name"] = f"inference-{sanitized_filename}-{str(uuid.uuid4())[:4]}"
        finetuned_plot_path = os.path.join(f"gs://{OUTPUT_BUCKET}", "plot_data", f"{base_filename}_finetuned.jsonl")

        logging.info(f"  - Submitting inference job for {base_filename}: {job_options['job_name']}")
        p = beam.Pipeline(options=PipelineOptions.from_dictionary(job_options))
        create_inference_pipeline(p, item['input_data'], finetuned_plot_path, item['finetuned_model_gcs_path'])
        result = p.run()
        inference_jobs.append(result)
        logging.info(f"  - Job {job_options['job_name']} is running (ID: {result.job_id()}).")

    # --- STAGE 4: Wait for all inference jobs to complete ---
    if inference_jobs:
        logging.info(f"\n{'='*80}\nWaiting for all {len(inference_jobs)} inference jobs to complete...\n{'='*80}")
        for job_result in inference_jobs:
            job_result.wait_until_finish()
            logging.info(f"  - Inference job {job_result.job_id()} finished with state: {job_result.state}")

    logging.info("\n✅ All parallel experiment workflows completed.")
