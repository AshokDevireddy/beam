import apache_beam as beam
import google.generativeai as genai
import logging
import os
import re
import json
import numpy as np
from apache_beam.utils.timestamp import Timestamp
from dotenv import load_dotenv
from apache_beam.transforms.userstate import BagStateSpec

import apache_beam as beam
import json
import numpy as np

from apache_beam.coders.coders import PickleCoder

from apache_beam.transforms.userstate import BagStateSpec, ReadModifyWriteStateSpec, TimerSpec, on_timer


class CustomJsonEncoderForLLM(json.JSONEncoder):
    """Encodes special types like Timestamp and numpy objects into JSON."""
    def default(self, obj):
        if isinstance(obj, Timestamp):
            # Store as a dict with a special key for easy decoding
            return {'__timestamp__': True, 'micros': obj.micros}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def custom_json_decoder(dct):
    """Decodes a Timestamp object from our custom dict format."""
    if '__timestamp__' in dct:
        return Timestamp(micros=dct['micros'])
    return dct

class JsonCoderWithNumpyAndTimestamp(beam.coders.Coder):
    """A custom Beam Coder that handles JSON serialization for Timestamps and numpy types."""
    def encode(self, value):
        return json.dumps(value, cls=CustomJsonEncoderForLLM).encode('utf-8')

    def decode(self, encoded):
        return json.loads(encoded.decode('utf-8'), object_hook=custom_json_decoder)

    def is_deterministic(self):
        return True


# It's highly recommended to manage API keys via GCP Secret Manager
# and access them as environment variables in your Dataflow job.
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class LLMClassifierFn(beam.DoFn):
    """
    Takes an anomaly, formats a detailed prompt with surrounding context,
    calls the Gemini model to classify it, and routes the original data
    based on the model's decision.

    This DoFn is stateful, deferring anomalies that occur too close to
    the end of a window until a subsequent window provides enough context.
    """

    DEFERRED_ANOMALIES_STATE = BagStateSpec(
        'deferred_anomalies', coder=JsonCoderWithNumpyAndTimestamp())
    YIELD_BUFFER_STATE = ReadModifyWriteStateSpec('yield_buffer', PickleCoder())
    
    # <<< CHANGE: Define a timer and a state to track if it's set
    EXPIRY_TIMER = TimerSpec('expiry', beam.TimeDomain.WATERMARK)
    # <<< CHANGE: Add state to track the last yielded timestamp
    LAST_YIELDED_TIMESTAMP_STATE = ReadModifyWriteStateSpec('last_yielded_ts', PickleCoder())




    def __init__(self, secret, context_points=25, slide_interval=128, expected_interval_secs=1):
        self.context_points = context_points
        self._model = None
        self.secret = secret
        self.slide_interval = slide_interval
        self.expected_interval_micros = expected_interval_secs * 1_000_000

        self._last_window_data = None


    def setup(self):
        # Configure the generative model

        genai.configure(api_key=self.secret)
        logging.getLogger().setLevel(logging.INFO)


        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 256,
            "response_mime_type": "application/json",
        }
        # For a full list of safety settings, see the Gemini API documentation
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ]
        self._model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info("Gemini Model has been successfully initialized.")

    def _build_prompt(self, anomaly_data, context_before, context_after):
        mean_before = np.mean(context_before) if context_before.size > 0 else 0
        mean_after = np.mean(context_after) if context_after.size > 0 else 0
        std_before = np.std(context_before) if context_before.size > 0 else 0
        std_after = np.std(context_after) if context_after.size > 0 else 0

        return f"""
        You are an expert time-series analyst classifying an outlier from NYC taxi pickup data.
        Normal behavior includes daily and weekly cyclical patterns.

        **1. Outlier Context:**
        * **--> The Outlier:**
            * **Timestamp:** {Timestamp(micros=anomaly_data['timestamp'].micros)}
            * **Actual Value:** {anomaly_data['actual_value']:.2f}
            * **Predicted Value:** {anomaly_data['predicted_value']:.2f}
            * **Anomaly Upper Bound:** {anomaly_data['upper_bound']:.2f}
            * **Anomaly Lower Bound:** {anomaly_data['lower_bound']:.2f}

        **2. Data Surrounding the Outlier:**
        * **Data Before ({len(context_before)} points):** {np.round(context_before, 2).tolist()}
        * **Data After ({len(context_after)} points):** {np.round(context_after, 2).tolist()}

        **3. Statistical Context:**
        * **Mean Before:** {mean_before:.2f}
        * **Mean After:** {mean_after:.2f}
        * **Std. Dev. Before:** {std_before:.2f}
        * **Std. Dev. After:** {std_after:.2f}

        **4. Your Task:**

        **Step 1: Analyze the Evidence.** In a few sentences, describe the behavior of the data *after* the outlier. Does it quickly revert to the "Predicted Value" or the "Mean Before"? Or does it establish a new level, closer to the "Mean After"?

        **Step 2: Make a Decision.** Classify the outlier.
        * **REMOVE:** If it's a transient, one-off event. This is likely if the data after the outlier rapidly returns to the established pattern.
        * **KEEP:** If it signifies a sustained shift in the pattern that the model should learn from. This is likely if the `Mean After` has shifted significantly.

        **Step 3: Provide Final Output.** Respond with a single JSON object. Do not add any text outside the JSON block.

        {{
          "reasoning_steps": "Your analysis from Step 1 goes here.",
          "decision": "KEEP or REMOVE",
          "confidence_score": <A float between 0.0 and 1.0>
        }}
        """

    def process(self, element,
                deferred_anomalies=beam.DoFn.StateParam(DEFERRED_ANOMALIES_STATE),
                yield_buffer=beam.DoFn.StateParam(YIELD_BUFFER_STATE),
                expiry_timer=beam.DoFn.TimerParam(EXPIRY_TIMER)):

        key, data = element
        window_start_ts = data['window_start_ts']

        # Set a timer to fire based on the event time of the current element.
        # Each new element will push the timer forward. The timer will only
        # fire when a gap in the input stream occurs, allowing the buffer
        # to contain data from multiple consecutive sliding windows.
        # We set it far enough ahead to allow the next window's data to arrive.
        grace_period_secs = self.slide_interval * 2
        expiry_timer.set(window_start_ts + grace_period_secs)
        anomalies_in_window = data.get('anomalies', [])
        values_in_element = data.get('values_array', [])

        for anomaly in anomalies_in_window:
             deferred_anomalies.add(anomaly)

        buffer = yield_buffer.read() or {}
        for i, value in enumerate(values_in_element):
            point_timestamp = Timestamp(micros=window_start_ts.micros + (i * self.expected_interval_micros))
            buffer[point_timestamp] = value
        yield_buffer.write(buffer)

    @on_timer(EXPIRY_TIMER)
    def on_expiry_timer(
        self,
        deferred_anomalies=beam.DoFn.StateParam(DEFERRED_ANOMALIES_STATE),
        yield_buffer=beam.DoFn.StateParam(YIELD_BUFFER_STATE),
        # <<< CHANGE: Add the new state parameter here
        last_yielded_ts_state=beam.DoFn.StateParam(LAST_YIELDED_TIMESTAMP_STATE)):
        
        all_anomalies_to_consider = list(deferred_anomalies.read())
        buffered_points_map = yield_buffer.read() or {}
        
        if not buffered_points_map:
            return

        sorted_points = sorted(buffered_points_map.items())
        all_timestamps = [ts for ts, val in sorted_points]
        all_values = [val for ts, val in sorted_points]
        
        anomalies_to_process_now = []
        prompts_to_batch = []
        final_deferred = []

        for anomaly_data in all_anomalies_to_consider:
            anomaly_ts = anomaly_data['timestamp']
            try:
                idx_in_full_data = all_timestamps.index(anomaly_ts)
                
                if (idx_in_full_data + self.context_points) < len(all_values):
                    start_ctx = max(0, idx_in_full_data - self.context_points)
                    end_ctx = idx_in_full_data + self.context_points + 1
                    
                    context_before = np.array(all_values[start_ctx:idx_in_full_data])
                    context_after = np.array(all_values[idx_in_full_data + 1:end_ctx])
                    
                    anomaly_data['index_in_window'] = idx_in_full_data
                    prompt = self._build_prompt(anomaly_data, context_before, context_after)
                    prompts_to_batch.append(prompt)
                    anomalies_to_process_now.append(anomaly_data)
                else:
                    final_deferred.append(anomaly_data)
            except ValueError:
                 final_deferred.append(anomaly_data)

        if prompts_to_batch:
            try:
                logging.info(f"Sending a batch of {len(prompts_to_batch)} prompts to the LLM.")
                responses = self._model.generate_content(prompts_to_batch)
                for anomaly_data, response in zip(anomalies_to_process_now, responses):
                    try:
                        response_data = json.loads(response.text)
                        decision = response_data.get('decision', 'KEEP').strip().upper()
                        idx = anomaly_data['index_in_window']

                        if decision == 'REMOVE':
                            logging.warning(f"LLM decided to REMOVE anomaly at {anomaly_data['timestamp']}. Imputing value.")
                            all_values[idx] = anomaly_data['predicted_value']
                    except (json.JSONDecodeError, AttributeError) as e:
                        logging.error(f"Error processing LLM response for {anomaly_data['timestamp']}: {e}. Defaulting to KEEP.")
            except Exception as e:
                logging.error(f"Error calling LLM with a batch: {e}. Defaulting to KEEP for all.")
        
        # <<< CHANGE: New logic to yield only new data
        last_yielded_ts = last_yielded_ts_state.read()
        latest_ts_in_batch = None

        for i, (ts, original_val) in enumerate(sorted_points):
            # Only yield points that are newer than the last batch we yielded
            if last_yielded_ts is None or ts > last_yielded_ts:
                yield {
                    'timestamp': ts, 
                    'value': all_values[i]
                }
                latest_ts_in_batch = ts

        # After yielding, update the state with the latest timestamp from this batch
        if latest_ts_in_batch:
            last_yielded_ts_state.write(latest_ts_in_batch)

        # Prune the buffer. We need to keep enough historical data to serve
        # as `context_before` for the anomalies that we are re-deferring.
        if latest_ts_in_batch:
            all_buffered_points = yield_buffer.read() or {}
            
            # Find the earliest timestamp we need to keep. This will be
            # `context_points` before the last yielded point, ensuring
            # context is available for the next batch.
            try:
                last_yielded_index = all_timestamps.index(latest_ts_in_batch)
                context_start_index = max(0, last_yielded_index - self.context_points)
                context_start_ts = all_timestamps[context_start_index]

                pruned_buffer = {
                    ts: val
                    for ts, val in all_buffered_points.items()
                    if ts >= context_start_ts
                }
                yield_buffer.write(pruned_buffer)
            except ValueError:
                # This can happen if the buffer is in an inconsistent state.
                # As a fallback, we clear it if we aren't deferring anything.
                logging.warning(
                    f"Could not find last yielded timestamp "
                    f"{latest_ts_in_batch} in buffer for pruning."
                )
                if not final_deferred:
                    yield_buffer.clear()
        elif not final_deferred:
            # If we didn't yield anything and we're not deferring anything,
            # the buffer is fully processed and can be cleared.
            yield_buffer.clear()

        # Re-add anomalies that couldn't be processed to the state so they can
        # be considered in the next firing.
        deferred_anomalies.clear()
        if final_deferred:
            logging.info(f"Re-deferring {len(final_deferred)} anomalies due to insufficient context.")
            for anomaly in final_deferred:
                deferred_anomalies.add(anomaly)


# import apache_beam as beam
# import google.generativeai as genai
# import logging
# import os
# import re
# import json
# import numpy as np
# from apache_beam.utils.timestamp import Timestamp

# class LLMClassifierFn(beam.DoFn):
#     """
#     Takes an anomaly, formats a detailed prompt with surrounding context,
#     calls the Gemini model to classify it, and routes the original data
#     based on the model's decision.
#     """

#     class Outputs:
#         KEPT_ANOMALY = "kept_anomaly"
#         REMOVED_ANOMALY = "removed_anomaly"
#         # Output all original points, anomalous or not, for reconstruction
#         ALL_POINTS = "all_points"

#     def __init__(self, secret, context_points=25, slide_interval=128):
#         self.context_points = context_points
#         self._model = None
#         self.secret = secret
#         self.slide_interval = slide_interval


#     def setup(self):
#         genai.configure(api_key=self.secret)
#         # logging.getLogger().setLevel(logging.INFO)


#         generation_config = {
#             "temperature": 0.2,
#             "top_p": 1,
#             "top_k": 1,
#             "max_output_tokens": 256,
#             "response_mime_type": "application/json",
#         }
#         # For a full list of safety settings, see the Gemini API documentation
#         safety_settings = [
#             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#         ]
#         self._model = genai.GenerativeModel(
#             model_name="gemini-1.5-flash-latest",
#             generation_config=generation_config,
#             safety_settings=safety_settings
#         )
#         logging.info("Gemini Model has been successfully initialized.")

#     def _build_prompt(self, anomaly_data, context_before, context_after):
#         mean_before = np.mean(context_before) if context_before.size > 0 else 0
#         mean_after = np.mean(context_after) if context_after.size > 0 else 0
#         std_before = np.std(context_before) if context_before.size > 0 else 0
#         std_after = np.std(context_after) if context_after.size > 0 else 0

#         return f"""
#         You are an expert time-series analyst classifying an outlier from NYC taxi pickup data.
#         Normal behavior includes daily and weekly cyclical patterns.

#         **1. Outlier Context:**
#         * **--> The Outlier:**
#             * **Timestamp:** {Timestamp(micros=anomaly_data['timestamp'].micros)}
#             * **Actual Value:** {anomaly_data['actual_value']:.2f}
#             * **Predicted Value:** {anomaly_data['predicted_value']:.2f}
#             * **Anomaly Upper Bound:** {anomaly_data['upper_bound']:.2f}
#             * **Anomaly Lower Bound:** {anomaly_data['lower_bound']:.2f}

#         **2. Data Surrounding the Outlier:**
#         * **Data Before ({len(context_before)} points):** {np.round(context_before, 2).tolist()}
#         * **Data After ({len(context_after)} points):** {np.round(context_after, 2).tolist()}

#         **3. Statistical Context:**
#         * **Mean Before:** {mean_before:.2f}
#         * **Mean After:** {mean_after:.2f}
#         * **Std. Dev. Before:** {std_before:.2f}
#         * **Std. Dev. After:** {std_after:.2f}

#         **4. Your Task:**

#         **Step 1: Analyze the Evidence.** In a few sentences, describe the behavior of the data *after* the outlier. Does it quickly revert to the "Predicted Value" or the "Mean Before"? Or does it establish a new level, closer to the "Mean After"?

#         **Step 2: Make a Decision.** Classify the outlier.
#         * **REMOVE:** If it's a transient, one-off event. This is likely if the data after the outlier rapidly returns to the established pattern.
#         * **KEEP:** If it signifies a sustained shift in the pattern that the model should learn from. This is likely if the `Mean After` has shifted significantly.

#         **Step 3: Provide Final Output.** Respond with a single JSON object. Do not add any text outside the JSON block.

#         {{
#           "reasoning_steps": "Your analysis from Step 1 goes here.",
#           "decision": "KEEP or REMOVE",
#           "confidence_score": <A float between 0.0 and 1.0>
#         }}
#         """

#     def process(self, element):
#         # This function now yields a single, complete, and cleaned window
#         logging.info(f"Processing element: {element}")
#         key, data = element
#         window_start_ts = data['window_start_ts']
#         anomalies_in_window = data.get('anomalies', [])
#         cleaned_values = list(data['values_array'])

#         logging.info(f"LLMClassifierFn received element for key {key}. Number of anomalies: {len(anomalies_in_window)}")

#         # Process each anomaly with the LLM
#         if anomalies_in_window:

#           prompts_to_batch = []

#           for anomaly_data in anomalies_in_window:
#               logging.info(f"Processing anomaly: {anomaly_data}")
#               idx = anomaly_data['index_in_window']

#               # ... (context extraction and prompt building are the same) ...
#               start_ctx = max(0, idx - self.context_points)
#               end_ctx = min(len(cleaned_values), idx + self.context_points + 1)
#               context_before = np.array(cleaned_values[start_ctx:idx])
#               context_after = np.array(cleaned_values[idx + 1:end_ctx])
#               prompt = self._build_prompt(anomaly_data, context_before, context_after)
#               prompts_to_batch.append(prompt)

#           try:
#               # 2. Make a single, batched API call.
#               logging.info(f"Sending a batch of {len(prompts_to_batch)} prompts to the LLM.")
#               responses = self._model.generate_content(prompts_to_batch)

#               # 3. Process the batch of responses.
#               for anomaly_data, response in zip(anomalies_in_window, responses):
#                   response_data = json.loads(response.text)
#                   decision = response_data.get('decision', 'KEEP').strip().upper()
#                   idx = anomaly_data['index_in_window']

#                   logging.info(f"LLM response for index {idx}: {response_data}")

#                   if decision == 'REMOVE':
#                       logging.warning(f"LLM decided to REMOVE anomaly at {anomaly_data['timestamp']}. Imputing value.")
#                       predicted_value = anomaly_data['predicted_value']
#                       cleaned_values[idx] = predicted_value
#                   else: # KEEP
#                       logging.info(f"LLM decided to KEEP anomaly at {anomaly_data['timestamp']}.")

#           except Exception as e:
#               logging.error(f"Error calling LLM with a batch: {e}. Defaulting to KEEP for all anomalies in this window.")
#                   # On error, we also do nothing and keep the original value.

#         # After iterating through all anomalies, yield the cleaned data points
#         # for this window. This ensures the data sent to the finetuning component is continuous.
#         # Yield only the new data points from the sliding window
#         start_index_of_new_data = len(cleaned_values) - self.slide_interval
#         for i in range(start_index_of_new_data, len(cleaned_values)):
#             # Calculate the correct timestamp for each point
#             point_timestamp = window_start_ts + i
#             yield {'timestamp': point_timestamp, 'value': cleaned_values[i]}
