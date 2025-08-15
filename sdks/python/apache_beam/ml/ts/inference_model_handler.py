import apache_beam as beam
from apache_beam.ml.inference.base import ModelHandler
import timesfm
import logging
import numpy as np
import os
from google.cloud import storage
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.utils.timestamp import Timestamp

class LatestModelCheckpointLoader(beam.PTransform):
    """A PTransform that finds the latest model checkpoint in a GCS path."""
    def __init__(self, gcs_bucket, gcs_prefix):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix

    def expand(self, pcoll):
        return pcoll | "FindLatestModel" >> beam.Map(self._find_latest_model_path)

    def _find_latest_model_path(self, _):
        try:
            storage_client = storage.Client()
            blobs = storage_client.list_blobs(self.gcs_bucket, prefix=self.gcs_prefix)
            # Filter for model files and find the most recent one
            model_blobs = [b for b in blobs if b.name.endswith(".pth")]
            latest_blob = max(model_blobs, key=lambda b: b.time_created, default=None)
            
            if latest_blob:
                path = f"gs://{self.gcs_bucket}/{latest_blob.name}"
                logging.info(f"Found latest finetuned model at: {path}")
                return path
        except Exception as e:
            logging.error(f"Error finding latest model in GCS: {e}")
        
        # Return a path to the base model if no finetuned one exists or an error occurs
        base_model = "google/timesfm-1.0-200m-pytorch"
        logging.info(f"No finetuned model found. Using base model: {base_model}")
        return base_model

class DynamicTimesFmModelHandler(ModelHandler[np.ndarray, np.ndarray, timesfm.TimesFm]):
    """
    A model handler that loads a TimesFM model from a dynamic path (GCS or Hugging Face).
    The model path is provided as a side input to RunInference.
    """
    def __init__(self, model_uri: str, hparams):
        self._hparams = hparams
        self._model = None
        self._model_uri = model_uri
        self._context_len = hparams.context_len
        self._horizon_len = hparams.horizon_len

    def load_model(self) -> timesfm.TimesFm:
        """Loads a model from the handler's current model_uri."""
        logging.info(f"Loading TimesFM model from path: {self._model_uri}...")
        
        checkpoint_config = {}
        if self._model_uri.startswith("gs://"):
            try:
                gcs = GcsIO()
                file_name = os.path.basename(self._model_uri)
                local_path = f"/tmp/{file_name}"
                with gcs.open(self._model_uri, 'rb') as f_in, open(local_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                checkpoint_config['path'] = local_path
                logging.info(f"Downloaded model from GCS to {local_path}")
            except Exception as e:
                logging.error(f"Failed to download model from GCS: {e}. Check path and permissions.")
                raise e # Re-raise the exception to fail fast if the model can't be loaded.
        else:
            checkpoint_config['huggingface_repo_id'] = self._model_uri

        self._model = timesfm.TimesFm(
            hparams=self._hparams,
            checkpoint=timesfm.TimesFmCheckpoint(**checkpoint_config)
        )
        logging.info("TimesFM model loaded successfully.")
        return self._model
    
    def update_model_path(self, model_path: str):
        """
        This method is called by RunInference when a new model metadata is available
        from the side input. It updates the model URI that `load_model` will use.
        """
        if not model_path:
            logging.info("Received an empty model path update. No action taken.")
            return
        logging.info(f"Received model update. New model URI: {model_path}")
        self._model_uri = model_path
        self._model = self.load_model()
        logging.info("Model has been updated in the handler.")

    def run_inference(self, batch, model, inference_args=None):
        """
            Runs inference on a batch of data.
            
            Note: While this is a standard method for ModelHandler, we will call the
            model's `forecast` method directly in our DoFn for clarity.
            """
        # print("Running inference on batch:", batch)
        # logging.info(f"Running inference on batch:", batch)

        anomalies_found = []

        key, (window_start_ts, _, values_array) = batch[0]

        # A window must have enough data for both context and horizon.
        # if len(values_array) < self.context_len + self.horizon_len:
        #     return

        current_context = np.array(values_array[:self._context_len])
        actual_horizon_values = np.array(
            values_array[self._context_len:self._context_len + self._horizon_len])

        print("Current context shape:", current_context.shape)
        print("Actual horizon values shape:", actual_horizon_values.shape)
        point_forecast, experimental_quantile_forecast = model.forecast(
            [current_context],
            freq=[0],
        )

        current_predicted_horizon_values = point_forecast[
            0, :, 0] if point_forecast.ndim == 3 else point_forecast[0]

        current_q20_values = experimental_quantile_forecast[0, :, 2]
        current_q30_values = experimental_quantile_forecast[0, :, 3]
        current_q70_values = experimental_quantile_forecast[0, :, 7]
        current_q80_values = experimental_quantile_forecast[0, :, 8]

        for j in range(len(actual_horizon_values)):
            current_actual = actual_horizon_values[j]

            point_Q1 = np.nanmean([current_q20_values[j], current_q30_values[j]])
            point_Q3 = np.nanmean([current_q70_values[j], current_q80_values[j]])
            point_IQR = point_Q3 - point_Q1

            upper_thresh = point_Q3 + 1.5 * point_IQR
            lower_thresh = point_Q1 - 1.5 * point_IQR

            logging.info(f"Comparing: current_actual={current_actual} (type: {type(current_actual)}), "
                         f"lower_thresh={lower_thresh} (type: {type(lower_thresh)}), "
                         f"upper_thresh={upper_thresh} (type: {type(upper_thresh)})")

            if current_actual > upper_thresh or current_actual < lower_thresh:
                score = (current_actual - upper_thresh
                            ) / point_IQR if current_actual > upper_thresh else (
                                lower_thresh - current_actual) / point_IQR

                anomaly_timestamp_seconds = (window_start_ts.micros / 1e6) + (
                    self._context_len + j)
                
                index_in_window = self._context_len + j

                anomalies_found.append({
                    'key': key,
                    'timestamp': Timestamp(anomaly_timestamp_seconds),
                    'index_in_window': index_in_window,
                    'actual_value': current_actual,
                    'predicted_value': current_predicted_horizon_values[j],
                    'is_anomaly': True,
                    'outlier_score': score,
                    'lower_bound': lower_thresh,
                    'upper_bound': upper_thresh,
                })
        result_with_context = (batch[0], anomalies_found)

        return [result_with_context]
