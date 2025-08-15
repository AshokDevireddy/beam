import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.pvalue import AsSingleton
import logging
import os
import timesfm

# Import your custom modules
from apache_beam.ml.ts.inference_model_handler import DynamicTimesFmModelHandler, LatestModelCheckpointLoader
from apache_beam.ml.ts.ordered_sliding_window import OrderedSlidingWindowFn, FillGapsFn

# --- Pipeline Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "your-gcp-project-id")
REGION = os.environ.get("GCP_REGION", "us-central1")
TEMP_LOCATION = "gs://your-bucket/temp"
STAGING_LOCATION = "gs://your-bucket/staging"
INPUT_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/your-production-data-subscription"
OUTPUT_TOPIC = f"projects/{PROJECT_ID}/topics/anomaly-alerts"
FINETUNED_MODEL_BUCKET = "your-finetuned-models-bucket"
FINETUNED_MODEL_PREFIX = "timesfm/checkpoints"

# --- Model & Window Parameters ---
CONTEXT_LEN = 512
HORIZON_LEN = 128
WINDOW_SIZE = CONTEXT_LEN + HORIZON_LEN
SLIDE_INTERVAL = HORIZON_LEN
EXPECTED_INTERVAL = 1
MODEL_CHECK_INTERVAL_SECONDS = 300  # Check for a new model every 5 minutes

def run_inference_pipeline(argv=None):
    pipeline_options = PipelineOptions(
        argv,
        streaming=True,
        project=PROJECT_ID,
        region=REGION,
        temp_location=TEMP_LOCATION,
        staging_location=STAGING_LOCATION,
        # runner="DataflowRunner",
        # requirements_file="requirements.txt",
    )
    
    # HParams should match the model being loaded
    hparams = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=HORIZON_LEN,
        context_len=CONTEXT_LEN,
    )
    model_handler = DynamicTimesFmModelHandler(hparams=hparams)

    with beam.Pipeline(options=pipeline_options) as p:
        # Side input: Periodically check for the latest model checkpoint in GCS
        latest_model_path_side_input = (
            p
            | "PeriodicModelCheck" >> PeriodicImpulse(
                fire_interval=MODEL_CHECK_INTERVAL_SECONDS, apply_windowing=False)
            | "GetLatestModelPath" >> LatestModelCheckpointLoader(
                gcs_bucket=FINETUNED_MODEL_BUCKET,
                gcs_prefix=FINETUNED_MODEL_PREFIX
              )
        )

        # Main pipeline: Read data, window, and run inference
        raw_data = (
            p
            | "ReadProdData" >> beam.io.ReadFromPubSub(subscription=INPUT_SUBSCRIPTION)
            # ... (same parsing and timestamping as the finetuning pipeline) ...
            | "Decode" >> beam.Map(lambda x: x.decode('utf-8'))
            | "Parse" >> beam.Map(lambda x: (int(x.split(',')[0]), float(x.split(',')[1])))
            | "AddTimestamp" >> beam.MapTuple(lambda ts, val: beam.window.TimestampedValue(val, ts))
            | "AddKey" >> beam.WithKeys(lambda _: "prod_taxi_series")
        )

        windowed_data = (
            raw_data
            | "ApplySlidingWindow" >> beam.ParDo(
                OrderedSlidingWindowFn(window_size=WINDOW_SIZE, slide_interval=SLIDE_INTERVAL))
            | "FillGaps" >> beam.ParDo(FillGapsFn(expected_interval=EXPECTED_INTERVAL))
            | "MapToDict" >> beam.Map(lambda x: {
                'key': x[0], 
                'window_start_ts': x[1][0], 
                'values_array': x[1][2]
              })
        )
        
        anomalies = (
            windowed_data
            | "RunDynamicInference" >> beam.ml.inference.base.RunInference(
                model_handler=model_handler,
                model_path=AsSingleton(latest_model_path_side_input)
              )
        )

        # Output the detected anomalies for alerting or further processing
        (anomalies
         | "LogAnomalies" >> beam.Map(logging.info)
         # Optional: Encode to JSON and write to a Pub/Sub topic for alerting
         # | "EncodeToJson" >> beam.Map(json.dumps)
         # | "WriteToAlertsTopic" >> beam.io.WriteToPubSub(topic=OUTPUT_TOPIC)
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_inference_pipeline()