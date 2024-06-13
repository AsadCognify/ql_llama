import json
from flask import Flask, request, jsonify
from qlm.quantum_processing import generate_prediction_from_llama3

app = Flask(__name__)


@app.route("/inference/llama3", methods=["POST"])
def inference_llama3():
    user_query = request.get_json()

    print(user_query)
    # epoch
    # query
    # prompt
    # storage_id
    # bot_name

    # Read the current value from the file
    tries = -1
    with open("tries.txt", 'r') as file:
        tries = int(file.read().strip())

    print(f"tries: {tries}")

    prediction = generate_prediction_from_llama3(
        epoch=user_query.get("epoch"), 
        bot_name=user_query.get("bot_name"),
        storage_id=user_query.get("storage_id"), 
        query=user_query.get("query"),
        system_prompt=user_query.get("prompt"),
        # use_base_model # finetuned inference (False) or base model inference
        tries = tries
    )

    tries += 1
    # Write the new value back to the file
    with open("tries.txt", 'w') as file:
        file.write(str(tries))
        
    # prediction = None
    print(f"prediction: {prediction}")

    return jsonify({"prediction": prediction.split("assistant\n\n")[1]}), 200
    # return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    print(__name__)