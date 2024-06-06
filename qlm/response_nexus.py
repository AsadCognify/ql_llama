from flask import jsonify, request
import logging
import time
import asyncio

logger = logging.getLogger('my_custom_logger')

def handle_incoming_request(status: str = None):
    # You don't necessarily need to process the request body here.
    # But if you want to access the data, use request.json
    # data = request.json
    data = request.get_json()
    # id = data.get("id")
    # email = data.get("email")

    # return jsonify({"id": id, "email": email, "status": status}), 200
    return jsonify(data), 200
    # Just return a success message and status code
    # return "Success!", 200

def pod_async_call():
  print("Your POD is running!")
  
  time.sleep(5)
  # await asyncio.sleep(5)

  print("Pod async call finished")
  
  return "Pod is running"

async def initialize_finetune_data(finetune_combination_list, combination_set, org):
    """
    Calls AWS for finetuning

    """
    try:
        for combination in finetune_combination_list:
            if combination.status == "Completed":
                logger.info("Already finetuned")
                continue
            elif combination.status == "In progress":
                logger.info("Finetuning is already in progress")
                continue
            else:
                data = {
                    "definition": {
                        "id": org.storage_id,
                        "action": "finetune",
                        "llm": "llama3",
                        "get_from": "s3",
                    },
                    "qlora_params": {
                        "Rank": combination.rank,
                        "Alpha": combination.alpha,
                    },
                    "general_ft_params": {
                        "batch_size": combination.batchSize,
                        "lr": combination.learningRate,
                        "epochs": combination.epoch,
                    },
                    "training_material": {
                        "training_dataset": combination_set.finetuning_dataset[0].train_file_name,
                        "validation_dataset": combination_set.finetuning_dataset[0].testing_file_name,
                    },
                }

                file_path = 'config.yaml'

                # Writing data to a YAML file
                # with open(file_path, 'w') as file:
                #     yaml.dump(data, file, default_flow_style=False)

                # Make an asynchronous call to AWS here
                # await aws_async_call()
                
                logger.info("Finetuning has started")

        return True
    except Exception as e:
        logger.error("Error Initilazing finetune enviorment because :", e)
        return False