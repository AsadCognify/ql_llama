import boto3
import os

def download_datasets(training_dataset: str, validation_dataset: str, bucket_name: str, folder_name: str, local_dir: str):
    """
    Download training and validation datasets from S3 bucket.

    :param training_dataset: Name of the training dataset file.
    :param validation_dataset: Name of the validation dataset file.
    :param bucket_name: Name of the S3 bucket.
    :param folder_name: Name of the folder in the S3 bucket.
    :param local_dir: Local directory to save the downloaded files.
    """
    s3_client = boto3.client('s3')

    def download_file_from_s3(file_name, folder_name):
        s3_path = f"{folder_name}/{file_name}"
        local_file_path = os.path.join(local_dir, file_name)
        print(
            f"local_file_path = {local_file_path}, file_name = {file_name}, bucket_name = {bucket_name}, folder_name = {folder_name}, s3_path = {s3_path}"
        )
        s3_client.download_file(bucket_name, s3_path, local_file_path)
        print(f"Downloaded {s3_path} to {local_file_path}")

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Download the files
    download_file_from_s3(training_dataset, folder_name)
    download_file_from_s3(validation_dataset, folder_name)

# Example usage
if __name__ == "__main__":
    training_material = {
        "training_dataset": "training_dataset.json",
        "validation_dataset": "validation_dataset.json"
    }
    bucket_name = "training_bucket"
    folder_name = "obscsr-medbpot-002"
    local_directory = "./datasets"

    download_datasets(
        training_material["training_dataset"],
        training_material["validation_dataset"],
        bucket_name,
        folder_name,
        local_directory
    )