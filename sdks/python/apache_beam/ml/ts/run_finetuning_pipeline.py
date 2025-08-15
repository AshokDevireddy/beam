import apache_beam as beam
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pvalue import AsDict
from apache_beam.ml.ts.ordered_sliding_window import OrderedSlidingWindowFn, FillGapsFn
import logging
import os
import timesfm
from apache_beam.utils.timestamp import Timestamp
import csv

# Import your custom modules
from apache_beam.ml.ts.llm_classifier import LLMClassifierFn
from apache_beam.ml.ts.finetuning_components import BatchAndTriggerFinetuningFn, RunFinetuningFn
from apache_beam.ml.ts.inference_model_handler import DynamicTimesFmModelHandler

# --- Pipeline Configuration ---
# These should be passed as pipeline arguments in a production environment
PROJECT_ID = os.environ.get("GCP_PROJECT", "apache-beam-testing")
REGION = os.environ.get("GCP_REGION", "us-central1")
TEMP_LOCATION = "gs://ashok-testing/temp"
STAGING_LOCATION = "gs://ashok-testing/staging"
INPUT_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/your-data-subscription"
FINETUNED_MODEL_BUCKET = "finetuned-models"
FINETUNED_MODEL_PREFIX = "timesfm/checkpoints"

# --- Model & Finetuning Parameters ---
BASE_MODEL_REPO = "google/timesfm-1.0-200m-pytorch"
CONTEXT_LEN = 512
HORIZON_LEN = 128
WINDOW_SIZE = CONTEXT_LEN + HORIZON_LEN
SLIDE_INTERVAL = HORIZON_LEN
EXPECTED_INTERVAL = 1  # Seconds between data points
FINETUNING_BATCH_SIZE = 10000  # Number of clean points to collect before finetuning
FINETUNE_CONFIG = {"learning_rate": 1e-5, "epochs": 3, "batch_size": 16}

def run_finetuning_pipeline(argv=None):
    options = PipelineOptions([
        "--streaming",
        "--environment_type=LOOPBACK",
        "--runner=PrismRunner",
    ])

    # options = PipelineOptions([
    #     "--streaming",
    #     "--runner=DataflowRunner",
    #     "--temp_location=gs://ashok-testing/anomaly-temp",
    #     "--staging_location=gs://ashok-testing/anomaly-temp",
    #     "--project=apache-beam-testing",
    #     "--region=us-central1",
    #     "--sdk_location=dist/apache_beam-2.67.0.dev0.tar.gz",
    #     "--requirements_file=/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/requirements.txt"
    # ])

    # HParams for the initial anomaly detection model
    hparams = timesfm.TimesFmHparams(
        backend="gpu",  # Or "cpu"
        per_core_batch_size=32,
        horizon_len=HORIZON_LEN,
        context_len=CONTEXT_LEN,
    )
    # Use the dynamic handler here as well, starting with the base model
    model_handler = DynamicTimesFmModelHandler(hparams=hparams)

    with open("apache_beam/ml/ts/nyc_taxi_timeseries.csv", "r") as f:
      reader = csv.reader(f)
      next(reader)  # Skip header row
      input_data = [(Timestamp(int(row[0])), float(row[2])) for row in reader]

    with beam.Pipeline(options=options) as p:
        # Step 1: Read from Pub/Sub and create sliding windows
        # raw_data = (
        #     p
        #     | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=INPUT_SUBSCRIPTION, with_attributes=False)
        #     | "Decode" >> beam.Map(lambda x: x.decode('utf-8'))
        #     | "Parse" >> beam.Map(lambda x: (int(x.split(',')[0]), float(x.split(',')[1])))
        #     | "AddTimestamp" >> beam.MapTuple(lambda ts, val: beam.window.TimestampedValue(val, ts))
        #     | "AddKey" >> beam.WithKeys(lambda _: "taxi_series")
        # )

        windowed_data = (
            p
            | PeriodicImpulse(data=input_data[:2000], fire_interval=0.01)
            | "AddKey" >> beam.WithKeys(lambda x: 0)       
            | "ApplySlidingWindow" >> beam.ParDo(
                OrderedSlidingWindowFn(window_size=WINDOW_SIZE, slide_interval=SLIDE_INTERVAL))
            | "FillGaps" >> beam.ParDo(FillGapsFn(expected_interval=EXPECTED_INTERVAL))
            | "Skip NaN Values for now" >> beam.Filter(
              lambda batch: 'NaN' not in batch[1][2])
            # | "MapToDict" >> beam.Map(lambda x: {
            #     'key': x[0], 
            #     'window_start_ts': x[1][0], 
            #     'values_array': x[1][2]
            #   })
        )

        # Step 2: Detect anomalies using the base TimesFM model
        # The model path is hardcoded here to the base model
        anomalies = (
            windowed_data
            | "DetectAnomalies" >> beam.ml.inference.base.RunInference(
                model_handler=model_handler,
                model_path=BASE_MODEL_REPO
              )
        )

        # Step 3: Group anomalies by their window key for joining
        anomalies_by_key = (
            anomalies
            | "KeyAnomalies" >> beam.Map(lambda x: (x['key'], x))
            | "GroupAnomalies" >> beam.GroupByKey()
        )

        # Step 4: Join windowed data with its detected anomalies
        def join_anomalies(element, anomalies_dict):
            key = element['key']
            element['anomalies'] = anomalies_dict.get(key, [])
            return key, element
        
        data_with_anomalies = (
            windowed_data
            | "AddKeyToWindow" >> beam.WithKeys(lambda x: x['key'])
            | "JoinAnomalies" >> beam.Map(join_anomalies, AsDict(anomalies_by_key))
        )

        # Step 5: Use LLM to classify anomalies and filter the dataset
        llm_results = (
            data_with_anomalies
            | "LLMClassifier" >> beam.ParDo(LLMClassifierFn()).with_outputs(
                LLMClassifierFn.Outputs.KEPT_ANOMALY,
                LLMClassifierFn.Outputs.REMOVED_ANOMALY,
                LLMClassifierFn.Outputs.ALL_POINTS
              )
        )
        
        clean_data_for_finetuning = llm_results[LLMClassifierFn.Outputs.ALL_POINTS]
        
        # Log removed anomalies for monitoring
        (llm_results[LLMClassifierFn.Outputs.REMOVED_ANOMALY] 
         | "LogRemoved" >> beam.Map(lambda x: logging.warning(f"REMOVED by LLM: {x}")))

        # Step 6: Batch the clean data to trigger a finetuning job
        finetuning_job_input = (
            clean_data_for_finetuning
            | "KeyForBatching" >> beam.WithKeys(lambda _: "finetune_batch")
            | "BatchForFinetuning" >> beam.ParDo(BatchAndTriggerFinetuningFn(FINETUNING_BATCH_SIZE))
        )

        # Step 7: Run the fine-tuning job and upload the new model to GCS
        (finetuning_job_input 
         | "RunFinetuning" >> beam.ParDo(
             RunFinetuningFn(
                 base_model_repo=BASE_MODEL_REPO,
                 finetuned_model_bucket=FINETUNED_MODEL_BUCKET,
                 finetuned_model_prefix=FINETUNED_MODEL_PREFIX,
                 hparams=hparams,
                 finetune_config=FINETUNE_CONFIG
             )
           )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_finetuning_pipeline()