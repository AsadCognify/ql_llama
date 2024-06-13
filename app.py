import json
import logging
from flask import Flask, request, jsonify
from qlm.quantum_processing import generate_prediction_from_llama3, load_model_and_tokenizer_for_llama3


# Basic configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
loaded_models = {}

## Route to load model to memory
@app.route("/load_model/llama3", methods=["POST"])
def load_model_llama3():
    """
    Load a Llama3 model and tokenizer for inference.

    This route handles a POST request to load a Llama3 model and tokenizer for inference. The request body should contain
    a JSON object with the following keys:
    - `storage_id`: The ID of the storage location where the model is stored.
    - `bot_name`: The name of the bot.
    - `epoch`: The epoch of the model.
    - `use_base_model` (optional): A boolean indicating whether to load the base model for inference (False) or the
      finetuned model (True).

    The function calls the `load_model_and_tokenizer_for_llama3` function with the provided parameters to load the model
    and tokenizer. It then stores the loaded model and tokenizer in the `loaded_models` dictionary using the storage ID
    and bot name as the key.

    Returns:
        A string indicating the success of the model loading with the format "Model loaded at {storage_id}.{bot_name}".
    """

    user_query = request.get_json()
    print(user_query)
    # epoch
    # storage_id
    # bot_name
    # (Optional) use_base_model # finetuned inference (False) or base model inference

    model, tokenizer = load_model_and_tokenizer_for_llama3(
        storage_id=user_query.get("storage_id"), 
        bot_name=user_query.get("bot_name"),
        epoch=user_query.get("epoch"),
        # load_base_model=user_query.get("use_base_model")
    )

    loaded_models[ f'{user_query.get("storage_id")}.{user_query.get("bot_name")}' ] = (model, tokenizer)

    return f'Model loaded at {user_query.get("storage_id")}.{user_query.get("bot_name")}'

## Route to run inference on a loaded model
@app.route("/inference/llama3", methods=["POST"])
def inference_llama3():
    user_query = request.get_json()

    print(user_query)
    # epoch
    # query
    # prompt
    # storage_id
    # bot_name

    model_name = f'{user_query.get("storage_id")}.{user_query.get("bot_name")}'

    if model_name in loaded_models:
        model, tokenizer = loaded_models[model_name]

        prediction = generate_prediction_from_llama3(
            model= model,
            tokenizer= tokenizer,
            query=user_query.get("query"),
            system_prompt=user_query.get("prompt")
        )

        # prediction = None
        logger.info(f"prediction: {prediction}")

        return jsonify({"prediction": prediction.split("assistant\n\n")[1]}), 200
        # return jsonify(prediction)

    else:
        return f'Model {user_query.get("storage_id")}.{user_query.get("bot_name")} not loaded. Please load the model first.', 400



# Start the flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    print(__name__)