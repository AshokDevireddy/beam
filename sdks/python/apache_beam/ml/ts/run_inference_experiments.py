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
from apache_beam.ml.ts.inference_model_handler import DynamicTimesFmModelHandler
from apache_beam.examples.inference.anomaly_detection.timesfm_anomaly_detection.ordered_sliding_window import OrderedSlidingWindowFn, FillGapsFn
from apache_beam.io.fileio import WriteToFiles, TextSink
from apache_beam.transforms import window

# =================================================================
# 1. SCRIPT CONFIGURATION
# =================================================================
GCP_PROJECT = "apache-beam-testing"
GCP_REGION = "us-central1"
INPUT_BUCKET = "ashok-testing"
INPUT_FOLDER = "times-series-datasets"
OUTPUT_BUCKET = "ashok-testing"
FINETUNED_MODEL_BUCKET = "ashok-testing"
FINETUNED_MODEL_PREFIX = "finetuned-models"
PLOT_DATA_BUCKET = "plot_data"
TEMP_LOCATION = f"gs://{OUTPUT_BUCKET}/temp"
STAGING_LOCATION = f"gs://{OUTPUT_BUCKET}/staging"
SETUP_FILE_PATH = "/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/setup.py"
INITIAL_MODEL = "google/timesfm-1.0-200m-pytorch"

# =================================================================
def sanitize_job_name_part(name_part):
    """
    Cleans a string to be a valid part of a Dataflow job name.
    It must only contain [-a-z0-9].
    """
    name_part = name_part.lower()
    sanitized = re.sub(r'[^a-z0-9-]+', '-', name_part)
    sanitized = sanitized.strip('-')
    if not sanitized:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return sanitized[:40]


def print_and_pass_through(label):
    def logger(element):
        print(f"--- {label} --- \nELEMENT: %s", element)
        return element
    return logger


class CustomJsonEncoder(json.JSONEncoder):
    """A custom JSON encoder that knows how to handle Beam's Timestamp objects."""
    def default(self, obj):
        if isinstance(obj, Timestamp):
            return obj.micros / 1e6
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

## =================================================================
## 3. MAIN ORCHESTRATION SCRIPT
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
        "dataflow_service_options" : "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver",
    }

    storage_client = storage.Client(project=GCP_PROJECT)

    dataset_blobs = storage_client.list_blobs(INPUT_BUCKET, prefix=INPUT_FOLDER)
    csv_files = [blob for blob in dataset_blobs if blob.name.endswith('.csv') and not blob.name.endswith('/')]

    if not csv_files:
        logging.error(f"No CSV files found in gs://{INPUT_BUCKET}/{INPUT_FOLDER}")

    jobs = []
    for blob in csv_files:
        file_uri = f"gs://{blob.bucket.name}/{blob.name}"
        base_filename = os.path.basename(blob.name).replace('.csv', '')
        logging.info(f"\n{'='*60}\n🚀 Preparing unified inference job for: {base_filename}\n{'='*60}")

        try:
            df = pd.read_csv(file_uri)
            data_values = df['Data'].to_numpy()
            input_data = [(Timestamp(i), value) for i, value in enumerate(data_values)]
            input_data = input_data[:8000]
            logging.info(f"  - Successfully loaded {len(input_data)} records.")

            # Add a final element with MAX_TIMESTAMP. This is crucial for
            # signaling the end of the stream to timer-based DoFns like
            # OrderedSlidingWindowFn. It ensures the watermark advances to
            # infinity, allowing all pending timers to fire and flush out any
            # buffered data.
            input_data.append((beam.utils.timestamp.MAX_TIMESTAMP, 0))
        except Exception as e:
            logging.error(f"  - Failed to load or parse {file_uri}. Skipping. Error: {e}")
            continue

        # --- Create a single pipeline for both models ---
        job_options = pipeline_options.copy()
        sanitized_filename = sanitize_job_name_part(base_filename)
        job_options["job_name"] = f"inference-unified-{sanitized_filename}-{str(uuid.uuid4())[:4]}"

        p = beam.Pipeline(options=PipelineOptions.from_dictionary(job_options))

        # --- Shared Input Processing ---
        CONTEXT_LEN, HORIZON_LEN = 512, 128
        WINDOW_SIZE, SLIDE_INTERVAL, EXPECTED_INTERVAL = CONTEXT_LEN + HORIZON_LEN, HORIZON_LEN, 1
        try:
            hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32, horizon_len=HORIZON_LEN, context_len=CONTEXT_LEN)
        except ImportError:
            hparams = {}

        raw_windowed_data = (
            p
            | 'PeriodicImpulse' >> PeriodicImpulse(
                data=input_data, fire_interval=0.001)
            | "AddKey" >> beam.WithKeys(lambda x: 0)
            | "ApplySlidingWindow" >> beam.ParDo(
                OrderedSlidingWindowFn(WINDOW_SIZE, SLIDE_INTERVAL))
            | "FillGaps" >> beam.ParDo(FillGapsFn(EXPECTED_INTERVAL))
            | "PrintWindowedData" >> beam.Map(
                print_and_pass_through("Windowed Data")))

        # The MAX_TIMESTAMP element we added creates a final, artificial window
        # to ensure all timers fire. We must filter this window out before
        # sending the data to the models for inference. The element `x` is a
        # tuple `(key, (window_start, window_end, values))`, so we filter based
        # on the window_end time.
        windowed_data = (
            raw_windowed_data
            | 'FilterOutFinalWindow' >> beam.Filter(
                lambda x: x[1][1] < beam.utils.timestamp.MAX_TIMESTAMP))

        # --- Branch 1: Original Model ---
        original_plot_path = f"gs://{OUTPUT_BUCKET}/{PLOT_DATA_BUCKET}/{base_filename}/original/"
        logging.info(f"  - Defining ORIGINAL model branch. Outputting to {original_plot_path}")
        original_inference = (
            windowed_data
            | "RunInferenceOriginal" >> RunInference(DynamicTimesFmModelHandler(model_uri=INITIAL_MODEL, hparams=hparams))
            | "PrintOriginalInference" >> beam.Map(print_and_pass_through("Original Inference"))
        )
        _ = (
            original_inference
            | "WindowOriginalOutput" >> beam.WindowInto(window.FixedWindows(10))
            | "FormatOriginalOutput" >> beam.Map(lambda x: json.dumps(x[1][1], cls=CustomJsonEncoder))
            | "WriteOriginal" >> WriteToFiles(path=original_plot_path, sink=TextSink())
        )

        # --- Branch 2: Fine-tuned Model (if available) ---
        model_dir_prefix = f"{FINETUNED_MODEL_PREFIX}/{base_filename}/"
        model_blobs = list(storage_client.list_blobs(FINETUNED_MODEL_BUCKET, prefix=model_dir_prefix))
        
        # Find the actual model file (e.g., ending in .pth)
        model_file_blob = next((blob for blob in model_blobs if blob.name.endswith('.pth')), None)

        if model_file_blob:
            finetuned_model_path = f"gs://{FINETUNED_MODEL_BUCKET}/{model_file_blob.name}"
            finetuned_plot_path = f"gs://{OUTPUT_BUCKET}/{PLOT_DATA_BUCKET}/{base_filename}/finetuned/"
            logging.info(f"  - Defining FINETUNED model branch using model from: {finetuned_model_path}. Outputting to {finetuned_plot_path}")

            finetuned_inference = (
                windowed_data
                | "RunInferenceFinetuned" >> RunInference(DynamicTimesFmModelHandler(model_uri=finetuned_model_path, hparams=hparams))
                | "PrintFinetunedInference" >> beam.Map(print_and_pass_through("Finetuned Inference"))
            )
            _ = (
                finetuned_inference
                | "WindowFinetunedOutput" >> beam.WindowInto(window.FixedWindows(10))
                | "FormatFinetunedOutput" >> beam.Map(lambda x: json.dumps(x[1][1], cls=CustomJsonEncoder))
                | "WriteFinetuned" >> WriteToFiles(path=finetuned_plot_path, sink=TextSink())
            )
        else:
            logging.warning(f"  - Could not find a fine-tuned model for {base_filename} in gs://{FINETUNED_MODEL_BUCKET}/{model_dir_prefix}. Skipping fine-tuned branch.")

        # --- Run the unified job ---
        logging.info(f"  - Submitting UNIFIED inference job: {job_options['job_name']}")
        result = p.run()
        jobs.append(result)
        logging.info(f"  - Job {job_options['job_name']} is running (ID: {result.job_id()}).")

    if jobs:
        logging.info(f"\n{'='*80}\nWaiting for all {len(jobs)} inference jobs to complete...\n{'='*80}")
        for job_result in jobs:
            try:
                job_result.wait_until_finish()
                logging.info(f"  - ✅ Job {job_result.job_id()} finished with state: {job_result.state}")
            except Exception as e:
                logging.error(f"  - ❌ Job {job_result.job_id()} FAILED. Error: {e}")

    logging.info("\n✅ All parallel inference workflows completed.")
