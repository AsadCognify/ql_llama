import json
import logging
from flask import Flask, request, jsonify
from qlm.quantum_processing import generate_prediction_from_llama3, load_model_and_tokenizer_for_llama3, clear_gpu_memory, clear_memory


# Basic configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
loaded_models = {}

@app.route("/load_model/llama3", methods=["POST"])
def load_model_llama3():
    """
    Load a Llama3 model and tokenizer for inference.
    """
    try:
        # Clear GPU memory before loading new model
        clear_gpu_memory(logger=logger, loaded_models=loaded_models)

        user_query = request.get_json()
        logger.debug(f"Received load model request: {user_query}")

        storage_id = user_query.get("storage_id")
        bot_name = user_query.get("bot_name")
        epoch = user_query.get("epoch")
        use_base_model = user_query.get("use_base_model", False)

        model, tokenizer = load_model_and_tokenizer_for_llama3(
            storage_id=storage_id, 
            bot_name=bot_name,
            epoch=epoch,
            load_base_model=use_base_model
        )

        model_key = f'{storage_id}.{bot_name}'
        loaded_models[model_key] = (model, tokenizer)
        logger.info(f"Model loaded at {model_key}")

        return jsonify({"message": f"Model loaded at {model_key}"}), 200
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/inference/llama3", methods=["POST"])
def inference_llama3():
    try:
        user_query = request.get_json()
        logger.debug(f"Received inference request: {user_query}")

        storage_id = user_query.get("storage_id")
        bot_name = user_query.get("bot_name")
        query = user_query.get("query")
        prompt = user_query.get("prompt")

        model_key = f'{storage_id}.{bot_name}'

        if model_key in loaded_models:
            model, tokenizer = loaded_models[model_key]
            prediction = generate_prediction_from_llama3(
                model=model,
                tokenizer=tokenizer,
                query=query,
                system_prompt=prompt
            )

            # Clear memory after generation
            clear_memory(logger=logger)

            logger.info(f"Prediction generated for {model_key}: {prediction}")
            return jsonify({"prediction": prediction.split("assistant\n\n")[1]}), 200
        else:
            logger.warning(f"Model {model_key} not loaded. Please load the model first.")
            return jsonify({"error": f"Model {model_key} not loaded. Please load the model first."}), 400
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 500



# Start the flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    print(__name__)