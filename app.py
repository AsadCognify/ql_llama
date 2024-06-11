from flask import Flask, request, jsonify
import concurrent.futures
from qlm.response_nexus import handle_incoming_request, pod_async_call
from qlm.quantum_processing import LLAMA3
from qlm.s3 import download_datasets

app = Flask(__name__)
executor = concurrent.futures.ThreadPoolExecutor()

@app.route("/finetune", methods=["POST"])
def handle_finetune_request():
    data = request.get_json()
    print(
        f"training_dataset = {data['definition']['storage_id']}/{data['training_material']['training_dataset']}\n\
        validation_dataset = {data['definition']['storage_id']}/{data['training_material']['validation_dataset']}\n\
        bucket_name = 'asadfinetunetesting'\n\
        local_dir = {data['definition']['combination_id']}"
    )

    print("Retrieving datasets...")
    download_datasets(
        training_dataset = f"{data['training_material']['training_dataset']}",
        validation_dataset=f"{data['training_material']['validation_dataset']}",
        # bucket_name="asadfinetunetesting", # For testing
        bucket_name = "queryloop-storage",
        folder_name=data["definition"]["storage_id"],
        local_dir=data["definition"]["combination_id"]
    )
    print("Retrieval complete!")
    
    # print(f"data = {data}")

    # Create the executor and submit the async task
    # Schedule the async call to run in the background
    # futures = executor.submit(LLAMA3.check_thread, data)
    futures = executor.submit(LLAMA3.finetune, data)
    print(futures)
    # executor.submit(pod_async_call)
    
    # Return immediately without waiting for the async task to complete
    # return handle_incoming_request()
    return jsonify({"Status": "Success!\n"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


"""
data = {
    "definition": {
        "storage_id": "obscsr-medbpot-002", # Folder of S3
        "combination_id": "com-88964-cust-445", # Local directory to save the downloaded files and finetuned weights
        "email": "a@c.h",
        "action": "finetune",
        "llm": "llama3",
        "get_from": "s3"
    },
    "qlora_params": {
        "Rank": 1,
        "Alpha": 0.5
    },
    "general_ft_params": {
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 10
    },
    "training_material": {
        "training_dataset": "name",
        "validation_dataset": "name"
    }
}
"""