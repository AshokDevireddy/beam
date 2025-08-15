# convert_model.py
import torch
import os

# --- Configuration ---
input_checkpoint_path = '/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/torch_model.ckpt'
output_path = '/Users/ashokrd/Developer/beam/sdks/python/apache_beam/ml/ts/timesfm_initial_model.pth'

print(f"Loading checkpoint from: {input_checkpoint_path}")

# Load the checkpoint. map_location='cpu' ensures you don't need a GPU.
checkpoint = torch.load(input_checkpoint_path, map_location=torch.device('cpu'))

# Checkpoints can store the weights (state_dict) directly or inside a key.
# This code handles both common cases.
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("Extracted state_dict from 'state_dict' key.")
else:
    state_dict = checkpoint
    print("Using the entire checkpoint file as the state_dict.")

print(f"Saving clean state_dict to: {output_path}")

# Save just the model weights to the new .pth file
torch.save(state_dict, output_path)

print("✅ Conversion complete.")