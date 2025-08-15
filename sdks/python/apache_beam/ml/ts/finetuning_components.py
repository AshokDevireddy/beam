import apache_beam as beam
import logging
import torch
import numpy as np
import timesfm
from os import path
from apache_beam.ml.ts.finetuning_torch import TimesFMFinetuner
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import PickleCoder
from huggingface_hub import snapshot_download
from apache_beam.io.gcp.gcsio import GcsIO # Add this import

from torch.utils.data import Dataset
from google.cloud import storage
from typing import Tuple


class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""
  def __init__(
      self,
      series: np.ndarray,
      context_length: int,
      horizon_length: int,
      freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    return x_context, input_padding, freq, x_future
  
class NoisyTimeSeriesDataset(TimeSeriesDataset):
    """
    An augmented version of TimeSeriesDataset that applies random Gaussian noise
    on-the-fly to each sample. This ensures different noise is used each epoch.
    """
    def __init__(self,
                 series: np.ndarray,
                 context_length: int,
                 horizon_length: int,
                 noise_level: float,
                 freq_type: int = 0):
        
        # Initialize the parent class to run _prepare_samples()
        super().__init__(series, context_length, horizon_length, freq_type)
        
        self.noise_level = noise_level
        self.data_std = np.std(self.series)
        self.noise_std = self.data_std * self.noise_level
        
        if self.data_std > 0:
            logging.info(
                f"NoisyTimeSeriesDataset initialized. "
                f"Data std: {self.data_std:.4f}, "
                f"Noise std: {self.noise_std:.4f} (level={self.noise_level})"
            )
        else:
            logging.warning("Data standard deviation is 0. Noise will not be effective.")

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Step 1: Get the raw numpy arrays from self.samples, just like the original class.
        x_context_np, x_future_np = self.samples[index]

        # Step 2: Add noise to the numpy arrays if enabled.
        if self.noise_level > 0 and self.noise_std > 0:
            context_noise = np.random.normal(0, self.noise_std, x_context_np.shape)
            future_noise = np.random.normal(0, self.noise_std, x_future_np.shape)
            
            x_context_np = x_context_np + context_noise
            x_future_np = x_future_np + future_noise

        # Step 3: Convert the (potentially noisy) numpy arrays to tensors.
        x_context = torch.tensor(x_context_np, dtype=torch.float32)
        x_future = torch.tensor(x_future_np, dtype=torch.float32)

        # Step 4: Create padding and frequency, exactly like the original class.
        input_padding = torch.zeros_like(x_context)
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        # Step 5: Return the final tensors. The structure is now identical.
        return x_context, input_padding, freq, x_future

def prepare_datasets(
    series: np.ndarray,
    context_length: int,
    horizon_length: int,
    freq_type: int = 0,
    train_split: float = 0.8,
    noise_level: float = 0.0) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(series) * train_split)
  train_data = series[:train_size]
  val_data = series[train_size:]

  # Create datasets with specified frequency type
  train_dataset = NoisyTimeSeriesDataset(
      train_data,
      context_length=context_length,
      horizon_length=horizon_length,
      noise_level=noise_level, # Pass the noise level
      freq_type=freq_type)

  val_dataset = TimeSeriesDataset(
      val_data,
      context_length=context_length,
      horizon_length=horizon_length,
      freq_type=freq_type)

  return train_dataset, val_dataset


class BatchContinuousAndOrderedFn(beam.DoFn):
    """
    A stateful DoFn that buffers elements, keeps them sorted, and emits
    a batch only when a full, continuous sequence of points is available.
    Includes detailed logging for debugging.
    """
    BUFFER_STATE = ReadModifyWriteStateSpec('buffer', PickleCoder())

    def __init__(self, batch_size, expected_interval_seconds=1):
        self.batch_size = batch_size
        self.interval = expected_interval_seconds
        # NEW LOGGING: Counter to avoid logging on every single element
        self.counter = 0

    def process(self, element, buffer=beam.DoFn.StateParam(BUFFER_STATE)):
        key, data = element
        timestamp = data['timestamp']
        value = data['value']

        # Increment the counter
        self.counter += 1

        current_buffer = buffer.read() or []
        current_buffer.append((timestamp, value))
        current_buffer.sort(key=lambda x: x[0])

        # NEW LOGGING: Periodically log the buffer status
        if self.counter % 100 == 0 and current_buffer:
            logging.info(
                f"Batching buffer now contains {len(current_buffer)} points. "
                f"Timestamps range from {current_buffer[0][0]} to {current_buffer[-1][0]}."
            )

        start_index = 0
        # A variable to track if we should continue searching from the start of the buffer
        search_from_start = True
        while search_from_start:
            search_from_start = False  # Assume we won't restart the search
            # Start searching for a continuous block from the beginning of the current buffer.
            for i in range(len(current_buffer) - self.batch_size + 1):
                # Check for continuity in the slice starting at `i`
                is_continuous = True
                for j in range(self.batch_size - 1):
                    ts1_seconds = current_buffer[i + j][0].seconds()
                    ts2_seconds = current_buffer[i + j + 1][0].seconds()
                    if ts2_seconds - ts1_seconds != self.interval:
                        is_continuous = False
                        break  # Gap found, this is not a valid starting point.
                
                if is_continuous:
                    # A continuous batch was found at index `i`.
                    logging.info(f"Continuous sequence found! Emitting batch of size {self.batch_size} starting at index {i}.")
                    
                    batch_to_yield = current_buffer[i : i + self.batch_size]
                    
                    # Log the timestamps of the batch being yielded
                    yielded_timestamps = [ts for ts, val in batch_to_yield]
                    logging.info(f"Yielding batch with timestamps from {yielded_timestamps[0]} to {yielded_timestamps[-1]}.")

                    formatted_batch = [{'timestamp': ts, 'value': val} for ts, val in batch_to_yield]
                    yield formatted_batch

                    # Remove the yielded batch from the buffer using `del` for in-place modification.
                    del current_buffer[i:i + self.batch_size]

                    # Log the state of the buffer after removal
                    if current_buffer:
                        logging.info(f"Buffer now contains {len(current_buffer)} elements, from {current_buffer[0][0]} to {current_buffer[-1][0]}.")
                    else:
                        logging.info("Buffer is now empty.")

                    # Since we've modified the buffer, we should restart the search from the beginning
                    search_from_start = True
                    break  # Exit the inner for-loop to restart the while-loop
            
            # If the for-loop completes without finding a continuous batch, `search_from_start` will be False,
            # and the while-loop will terminate.

        # Write the final state of the buffer.
        logging.info(f"Writing {len(current_buffer)} elements back to state.")
        buffer.write(current_buffer)

class RunFinetuningFn(beam.DoFn):
  """
    Takes a batch of data, loads the LATEST model, runs fine-tuning, 
    and uploads the new model to GCS. This DoFn is stateful to ensure
    that it runs only once per key.
  """

  def __init__(
      self,
      initial_model_path, # Renamed from base_model_path
      finetuned_model_bucket,
      finetuned_model_prefix,
      hparams,
      config,
      noise_level=0.0):
    # This is now a fallback for the very first run
    self.initial_model_path = initial_model_path 
    self.finetuned_model_bucket = finetuned_model_bucket
    self.finetuned_model_prefix = finetuned_model_prefix
    self.hparams = hparams
    self.config = config
    self._storage_client = None
    self.noise_level = noise_level

  def setup(self):
    self._storage_client = storage.Client()
  
  def _get_latest_model_from_gcs(self):
    """Directly queries GCS for the most recently created model checkpoint."""
    try:
        bucket = self._storage_client.get_bucket(self.finetuned_model_bucket)
        blobs = list(bucket.list_blobs(prefix=self.finetuned_model_prefix))
        
        # Filter for actual model files and exclude the initial model if present
        model_blobs = [b for b in blobs if b.name.endswith(".pth") and "initial" not in b.name]

        if not model_blobs:
            return None
        
        # Find the blob with the latest creation time
        latest_blob = max(model_blobs, key=lambda b: b.time_created)
        latest_model_path = f"gs://{self.finetuned_model_bucket}/{latest_blob.name}"
        return latest_model_path
    except Exception as e:
        logging.error(f"Error querying GCS for the latest model: {e}")
        return None

  # Add the side input parameter to the process method
  def process(self, batch_of_data):

    logging.info(
        f"Received a batch of {len(batch_of_data)} points for finetuning.")

    # If a finetuned model exists, use it. Otherwise, use the initial base model.
    latest_model_path = self._get_latest_model_from_gcs()
    
    if latest_model_path:
        model_to_load = latest_model_path
        logging.info(f"Continuously finetuning from latest model: {model_to_load}")
    else:
        model_to_load = self.initial_model_path
        logging.info(f"No finetuned model found. Starting from initial model: {model_to_load}")

    # The input is a list of dictionaries; access keys directly, not by index.
    batch_of_data.sort(key=lambda x: x['timestamp'].micros)
    time_series_values = np.array([d['value'] for d in batch_of_data],
                                  dtype=np.float32)
    train_dataset, val_dataset = prepare_datasets(
        series=time_series_values,
        context_length=self.hparams.context_len,
        horizon_length=self.hparams.horizon_len,
        freq_type=self.config.freq_type,
        train_split=0.8,
        noise_level=self.noise_level # Pass the configured noise level
    )

    # Load the model (base or latest finetuned)
    # The updated get_model function can handle both GCS and Hugging Face paths
    model = get_model(
        model_path=model_to_load, # Use the path we just determined
        hparams=self.hparams,
        load_weights=True
    )

    # 4. Run fine-tuning (same as before)
    finetuner = TimesFMFinetuner(model, self.config)
    finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)

    # 5. Save and upload the new model (same as before)
    from datetime import datetime
    timestamp_str = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_filename = f"timesfm_finetuned_{timestamp_str}.pth"
    local_path = f"/tmp/{model_filename}"
    torch.save(model.state_dict(), local_path)
    bucket = self._storage_client.bucket(self.finetuned_model_bucket)
    blob_path = f"{self.finetuned_model_prefix}/{model_filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    logging.info(
        f"Successfully uploaded new model to gs://{self.finetuned_model_bucket}/{blob_path}"
    )
    
    yield blob_path


def get_model(model_path: str, hparams: TimesFmHparams, load_weights: bool = False):
    """
    Loads a TimesFM model from either a Hugging Face repo ID or a GCS path.
    The `load_weights` argument is kept for signature consistency but is
    effectively always True, as TimesFm handles loading.
    """
    checkpoint_config = {}
    
    # Case 1: The model path is a GCS URI.
    # We download it to a local file and tell TimesFmCheckpoint to load from that path.
    if model_path.startswith("gs://"):
        logging.info(f"Preparing to load model from GCS path: {model_path}")
        local_temp_path = f"/tmp/{path.basename(model_path)}"
        with GcsIO().open(model_path, 'rb') as f_in, open(local_temp_path, 'wb') as f_out:
            f_out.write(f_in.read())
        # The key for a local file is 'path'
        checkpoint_config['path'] = local_temp_path
        
    # Case 2: The model path is a Hugging Face repository ID.
    else:
        logging.info(f"Preparing to load model from Hugging Face repo: {model_path}")
        # The key for a Hugging Face repo is 'huggingface_repo_id'
        checkpoint_config['huggingface_repo_id'] = model_path

    # Initialize the TimesFm object correctly with the dynamically created checkpoint config.
    # This single call handles model configuration and weight loading.
    tfm = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(**checkpoint_config)
    )

    logging.info("Model loaded successfully inside get_model.")
    
    # The `TimesFm` object holds the configured model instance.
    # The model returned here will be a PatchedTimeSeriesDecoder instance with weights loaded.
    return tfm._model
