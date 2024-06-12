from flask import Flask, request, jsonify
from qlm.quantum_processing import generate_prediction_from_llama3

app = Flask(__name__)


@app.route("/inference/llama3", methods=["POST"])
def inference_llama3():
    user_query = request.to_json()

    print(user_query)
    # epoch
    # query
    # prompt
    # storage_id
    # bot_name

    prediction = generate_prediction_from_llama3(
        epoch=user_query.get("epoch"), 
        bot_name=user_query.get("bot_name"),
        storage_id=user_query.get("storage_id"), 
        query=user_query.get("query"),
        system_prompt=user_query.get("prompt")
    )

    # prediction = None

    return jsonify({"input": user_query, "prediction": prediction})