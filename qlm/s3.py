import os
import boto3
import subprocess

def download_from_s3(s3_path: str, local_dir: str):
    """
    s3: complete s3 file path
    local_dir: where the downloaded files will be saved
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    try:
        s3_client = boto3.client('s3')
        bucket_name = "queryloop-storage"
        
        s3_client.download_file(bucket_name, s3_path, f"{local_dir}/{os.path.basename(s3_path)}")
        
        # s3_client.get_object(Bucket='examplebucket', Key=s3_path, f"{local_dir}/training_dataset.json")
        print(f"Downloaded {s3_path} to {local_dir}")
    
    except Exception as e:
        print(f"Exception occured: {e}")


def sync_from_s3(s3_folder: str, local_dir: str):
    try:
        bucket_name = "queryloop-storage"

        # Construct the command
        command = [
            'aws', 's3', 'sync',
            f's3://{bucket_name}/{s3_folder}',
            local_dir,
            # '--delete'
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("Sync successful")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("Sync failed with error:")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")