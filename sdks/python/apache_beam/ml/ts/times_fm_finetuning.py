import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pvalue import AsDict, AsSingleton
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.examples.inference.anomaly_detection.timesfm_anomaly_detection.ordered_sliding_window import OrderedSlidingWindowFn, FillGapsFn
import logging
import os
import json
import timesfm
from apache_beam.utils.timestamp import Timestamp
import csv
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.utils import WatchFilePattern
import typing
from apache_beam.ml.ts.finetuning_torch import FinetuningConfig

# Import your custom modules
from apache_beam.ml.ts.llm_classifier import LLMClassifierFn
from apache_beam.ml.ts.finetuning_components import BatchContinuousAndOrderedFn, RunFinetuningFn
from apache_beam.ml.ts.inference_model_handler import DynamicTimesFmModelHandler, LatestModelCheckpointLoader

# --- Pipeline Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "apache-beam-testing")
REGION = os.environ.get("GCP_REGION", "us-central1")
TEMP_LOCATION = "gs://ashok-testing/temp"
STAGING_LOCATION = "gs://ashok-testing/staging"
# INPUT_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/your-data-subscription"
# ANOMALY_ALERTS_TOPIC = f"projects/{PROJECT_ID}/topics/anomaly-alerts"
FINETUNED_MODEL_BUCKET = "ashok-testing"
FINETUNED_MODEL_PREFIX = "finetuned-models/timesfm/checkpoints"

# --- Model & Window Parameters ---
CONTEXT_LEN = 512
HORIZON_LEN = 128
WINDOW_SIZE = CONTEXT_LEN + HORIZON_LEN
SLIDE_INTERVAL = HORIZON_LEN
EXPECTED_INTERVAL = 1
INITIAL_MODEL = "google/timesfm-1.0-200m-pytorch"

MODEL_CHECK_INTERVAL_SECONDS = 10  # Check for a new model every 5 seconds
FINETUNING_BATCH_SIZE = WINDOW_SIZE * 2 # make larger later. minimum is WINDOW_SIZE for validation and training
FINETUNE_CONFIG = FinetuningConfig(
      batch_size=128,          
      num_epochs=5,            
      learning_rate=1e-4,
      use_wandb=False,         
      freq_type=0, # should change based on your data
      log_every_n_steps=10,
      val_check_interval=0.5,
      use_quantile_loss=True
  )


def run_unified_pipeline(argv=None):
    # options = PipelineOptions([
    #     "--streaming",
    #     "--environment_type=LOOPBACK",
    #     "--runner=PrismRunner",
    #     "--logging_level=INFO",
    # ])

    options = PipelineOptions([
        "--streaming",
        "--runner=DataflowRunner",
        "--temp_location=gs://ashok-testing/anomaly-temp",
        "--staging_location=gs://ashok-testing/anomaly-temp",
        "--project=apache-beam-testing",
        "--region=us-central1",
        "--sdk_location=dist/apache_beam-2.67.0.dev0.tar.gz",
        "--setup_file=/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/setup.py"
    ])

    with open("apache_beam/ml/ts/nyc_taxi_timeseries.csv", "r") as f:
      reader = csv.reader(f)
      next(reader)  # Skip header row
      input_data = [(Timestamp(int(row[0])), float(row[2])) for row in reader]

    # HParams for the model
    hparams = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=HORIZON_LEN,
        context_len=CONTEXT_LEN,
    )
    model_handler = DynamicTimesFmModelHandler(model_uri=INITIAL_MODEL, hparams=hparams)

    def print_and_pass_through(label):
        def logger(element):
            logging.info(f"--- {label} --- \nELEMENT: %s", element)
            return element
        return logger

    with beam.Pipeline(options=options) as p:
        # =================================================================
        # 1. Get Latest Model Path (Side Input)
        # =================================================================
        model_pattern = os.path.join(
            f"gs://{FINETUNED_MODEL_BUCKET}", FINETUNED_MODEL_PREFIX, "*.pth"
        )
        model_metadata_pcoll = (
            p
            | "WatchForNewModels" >> WatchFilePattern(
                file_pattern=model_pattern,
                interval=MODEL_CHECK_INTERVAL_SECONDS
              )
            | "PrintModelLocation" >> beam.Map(print_and_pass_through("Model Location"))

        )

        # =================================================================
        # 2. Ingest and Window Raw Data
        # =================================================================
        # raw_data = (
        #     p
        #     | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=INPUT_SUBSCRIPTION)
        #     | "Decode" >> beam.Map(lambda x: x.decode('utf-8'))
        #     | "Parse" >> beam.Map(lambda x: (int(x.split(',')[0]), float(x.split(',')[1])))
        #     | "AddTimestamp" >> beam.MapTuple(lambda ts, val: beam.window.TimestampedValue(val, ts))
        #     | "AddKey" >> beam.WithKeys(lambda _: "taxi_series")
        # )

        windowed_data = (
            # raw_data
            p
            | PeriodicImpulse(data=input_data[:3000], fire_interval=0.01)
            | "AddKey" >> beam.WithKeys(lambda x: 0)
            # | "PrintData" >> beam.Map(print_and_pass_through("Data"))
            | "ApplySlidingWindow" >> beam.ParDo(
                OrderedSlidingWindowFn(window_size=WINDOW_SIZE, slide_interval=SLIDE_INTERVAL))
            | "FillGaps" >> beam.ParDo(FillGapsFn(expected_interval=EXPECTED_INTERVAL)).with_output_types(
                typing.Tuple[int, typing.Tuple[Timestamp, Timestamp, typing.List[float]]])
            | "Skip NaN Values for now" >> beam.Filter(
              lambda batch: 'NaN' not in batch[1][2])
            # | "PrintFillGaps" >> beam.Map(print_and_pass_through("FILLGAPS"))
            # | "MapWindowToDict" >> beam.Map(lambda x: {
            #     'key': x[0],
            #     'window_start_ts': x[1][0],
            #     'values_array': x[1][2]
            #   })
            | "PrintWindowedData" >> beam.Map(print_and_pass_through("Windowed Data"))

        )

        # =================================================================
        # 3. Detect Anomalies using the Latest Model
        # =================================================================
        

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


        data_for_llm = (
            windowed_data
            # RunInference will now output PredictionResult(example, inference)
            # where 'inference' is the list of anomalies we created.
            | "DetectAnomalies" >> RunInference(
                model_handler=model_handler,
                # include_inputs=True, 
                model_metadata_pcoll=model_metadata_pcoll
              )
            | "PrintInference" >> beam.Map(print_and_pass_through("Inference Results"))
            | "FormatForLLM" >> beam.Map(format_for_llm)
            | "PrintDataForLLM" >> beam.Map(print_and_pass_through("Data for LLM"))
        )
        

        # =================================================================
        # 6. Classify with LLM and Create Clean Data for Finetuning
        # =================================================================

        clean_data_for_finetuning = (
            data_for_llm
            | "LLMClassifierAndImputer" >> beam.ParDo(
                LLMClassifierFn(secret=os.getenv("GEMINI_API_KEY"))
)
            # | "PrintCleanData" >> beam.Map(print_and_pass_through("Clean Data for Finetuning"))
        )


        # =================================================================
        # 7. Batch Clean Data and Trigger Finetuning
        # =================================================================
        finetuning_job_input = (
            clean_data_for_finetuning
            | "KeyForBatching" >> beam.WithKeys(lambda _: "finetune_batch")
            | "BatchAndTrigger" >> beam.ParDo(BatchContinuousAndOrderedFn(FINETUNING_BATCH_SIZE, 1))
            | "PrintFinetuningJobInput" >> beam.Map(print_and_pass_through("Finetuning Job Input"))
        )

        # =================================================================
        # 8. Run Finetuning and Save New Model to GCS
        # =================================================================
        (
            finetuning_job_input
            | "RunFinetuning" >> beam.ParDo(
                RunFinetuningFn(
                    initial_model_path="google/timesfm-1.0-200m-pytorch",
                    finetuned_model_bucket=FINETUNED_MODEL_BUCKET,
                    finetuned_model_prefix=FINETUNED_MODEL_PREFIX,
                    hparams=hparams,
                    config=FINETUNE_CONFIG
                ),
            )
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_unified_pipeline()