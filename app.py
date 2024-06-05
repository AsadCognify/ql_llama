from flask import Flask, request
from qlm.response_nexus import handle_incoming_request, pod_async_call
import concurrent.futures

app = Flask(__name__)
executor = concurrent.futures.ThreadPoolExecutor()

@app.route("/finetune", methods=["POST"])
def handle_finetune_request():
    # Schedule the async call to run in the background
    executor.submit(pod_async_call)
    
    # Return immediately without waiting for the async task to complete
    return handle_incoming_request()

if __name__ == "__main__":
    app.run(debug=True)
