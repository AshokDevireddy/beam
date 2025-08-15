import pandas as pd
import os
import random
from collections import defaultdict

def check_file_criteria(file_path):
    """
    Checks if a single CSV file meets the specified criteria.

    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        bool: True if the file meets all criteria, False otherwise.
    """
    try:
        # NOTE: Updated column names to match your latest code ('Data', 'Label')
        df = pd.read_csv(file_path, usecols=['data', 'label'])
    except (FileNotFoundError, ValueError, KeyError) as e:
        # Trying with original names as a fallback
        try:
            df = pd.read_csv(file_path, usecols=['Data', 'Label'])
            # Renaming for consistency within the function
            df.columns = ['data', 'label']
        except (FileNotFoundError, ValueError, KeyError) as e_fallback:
            print(f"Could not process {file_path}: {e_fallback}")
            return False

    # 1. Check for length
    # NOTE: Using your latest criteria from the provided code
    total_points = len(df)
    if total_points <= 8000 or total_points >= 70000:
        return False

    # Check if there are any anomalies at all
    if df['label'].sum() == 0:
        return False
        
    # 2. Check for anomalies in the last 20% of the data
    start_of_last_20_percent = total_points * 0.8
    anomaly_indices = df.index[df['label'] == 1]
    if anomaly_indices.max() < start_of_last_20_percent:
        return False

    # 3. Check for multiple, non-contiguous anomaly groups
    # NOTE: Using your latest criteria (anomaly_blocks < 4)
    anomaly_blocks = (df['label'].diff() == 1).sum()
    if anomaly_blocks < 4:
        return False
        
    return True

def find_and_group_datasets(root_folder):
    """
    Recursively finds valid CSVs and groups them by their parent subdirectory.

    Args:
        root_folder (str): The path to the main dataset folder.

    Returns:
        dict: A dictionary where keys are subdirectory names and values are lists
              of qualifying file paths.
    """
    grouped_files = defaultdict(list)
    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(dirpath, filename)
                if check_file_criteria(full_path):
                    subdir_name = os.path.basename(dirpath)
                    grouped_files[subdir_name].append(full_path)
                    
    return grouped_files

def select_final_datasets(grouped_files, sample_size=10):
    """
    Selects a final list of datasets based on the count per subdirectory.
    - If count < sample_size, take all.
    - If count >= sample_size, take a random sample.

    Args:
        grouped_files (dict): A dictionary with subdirectory names as keys and
                              lists of file paths as values.
        sample_size (int): The number of files to sample from larger groups.

    Returns:
        list: A final, sampled list of file paths.
    """
    final_list = []
    print("\n-------------------------------------------")
    print(f"📋 Selecting final datasets (take all if < {sample_size}, sample {sample_size} if not):")
    print("-------------------------------------------")

    # Sort for consistent output
    sorted_grouped_files = dict(sorted(grouped_files.items()))
    
    for subdir, file_list in sorted_grouped_files.items():
        if len(file_list) < sample_size:
            final_list.extend(file_list)
            print(f"'{subdir}': Taking all {len(file_list)} files.")
        else:
            sample = random.sample(file_list, sample_size)
            final_list.extend(sample)
            print(f"'{subdir}': Randomly sampling {sample_size} of {len(file_list)} files.")
            
    return final_list

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this path to the location of your 'TSB-UAD-Public-v2' folder
    main_folder_path = '/Users/ashokrd/Downloads/TSB-UAD-Public-v2' 
    
    print(f"🔍 Starting search in: {os.path.abspath(main_folder_path)}")
    
    # 1. Find all datasets that meet the criteria and group them
    grouped_datasets = find_and_group_datasets(main_folder_path)
    
    # Calculate total found before sampling
    total_found = sum(len(files) for files in grouped_datasets.values())

    print("\n-------------------------------------------")
    print(f"📊 Found {total_found} total datasets meeting the initial criteria.")
    print("-------------------------------------------")

    # Display the frequency counts for all found files
    print("📂 Initial Frequency per Subdirectory:")
    print("-------------------------------------------")
    if not grouped_datasets:
        print("No files found.")
    else:
        sorted_counts = dict(sorted(grouped_datasets.items()))
        for subdir, files in sorted_counts.items():
            print(f"{subdir}: {len(files)}")
    
    # 2. Select the final list based on the new sampling logic
    final_datasets_to_save = select_final_datasets(grouped_datasets, sample_size=10)
    
    # 3. Save the final, selected list to a file
    print("\n-------------------------------------------")
    print(f"💾 Saving {len(final_datasets_to_save)} selected files...")
    print("-------------------------------------------")
    
    if final_datasets_to_save:
        # NOTE: Using the file path from your code
        output_path = 'sdks/python/apache_beam/ml/ts/helper/filtered_datasets.txt'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for file_path in sorted(final_datasets_to_save): # Sort for reproducibility
                f.write(file_path + '\n')
        print(f"\n✅ Final selected datasets saved to '{output_path}'.")
    else:
        print("No datasets were selected.")