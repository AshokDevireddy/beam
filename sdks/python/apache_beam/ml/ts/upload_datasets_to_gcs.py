import os
import subprocess

def upload_datasets(dataset_list_path, gcs_bucket_url):
    """
    Reads a file containing a list of local file paths and uploads each file
    to a specified Google Cloud Storage bucket.

    Args:
        dataset_list_path (str): The path to the file containing the list of datasets.
        gcs_bucket_url (str): The GCS bucket URL where the files will be uploaded.
    """
    with open(dataset_list_path, 'r') as f:
        for line in f:
            local_path = line.strip()
            if not local_path:
                continue

            filename = os.path.basename(local_path)
            # Ensure the GCS path is correctly formatted with a trailing slash
            if not gcs_bucket_url.endswith('/'):
                gcs_bucket_url += '/'
            gcs_path = gcs_bucket_url + filename

            print(f"Uploading {local_path} to {gcs_path}...")
            try:
                # Using gsutil to copy the file
                subprocess.run(['gsutil', 'cp', local_path, gcs_path], check=True)
                print("Upload successful.")
            except subprocess.CalledProcessError as e:
                print(f"Error uploading {local_path}: {e}")
            except FileNotFoundError:
                print(f"gsutil command not found. Please ensure Google Cloud SDK is installed and in your PATH.")

if __name__ == "__main__":
    # The path to the file listing the datasets.
    datasets_file = 'apache_beam/ml/ts/helper/filtered_datasets.txt'
    
    # The destination GCS bucket URL.
    bucket_url = 'gs://ashok-testing/times-series-datasets/'
    
    upload_datasets(datasets_file, bucket_url)
