import os
import time
import requests
from typing import Dict, Any
from qlm.llama3.llama3_finetuning import loading_model_and_tokenizer, training_model, load_dataset_for_training
from qlm.llama3.finetuning_variables import LLAMA3TrainingConfig
from qlm.llama3.data_prep import Data_Prep
from qlm.s3 import sync_to_s3
class LLAMA3:
    def __init__(self) -> None:
        pass

    @staticmethod
    def check_thread(params: Dict[str, Any]):
        time.sleep(10)
        print(f"\
              \n\n-------------------------------------- FROM THE THREAD HERE --------------------------------------\n\n\
              {params}\
              \n\n-------------------------------------- FROM THE THREAD HERE --------------------------------------\n\n\
            ")

    @staticmethod
    def finetune(params: Dict[str, Any]):
        try:

            # Send finetune completion request to endpoint
            request_url = f'https://dev.queryloop-ai.com/api/eval_bot/set/finetune/status/{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}'
            print(request_url)
            payload = {"status": "running"}
            response = requests.post(request_url, json=payload)
            print(response.json())

            
            # Define variables
            ## Initalize config
            config = LLAMA3TrainingConfig()
            # print(f"\n\nFROM THE WORKER: Creating instance of LLAMA3TrainingConfig. {config}\n\n")
            LLAMA3.send_log_to_flask(message=f"\n---------------------------------- BEGIN OF TRANSMISSION --------------------------------------\nFROM THE WORKER: Creating instance of LLAMA3TrainingConfig. {config}")
            ## Manually set variables (Not the desired way)
            # config.set_variables(
            #     data_path="medmcq-736_out_of_182k_ready_to_train.json",
            #     model_dir="Meta-Llama-3-8B-Instruct",
            #     out_path="output/Llama3_finetuning/Llama3-8b_instruct_r=20_a_40_b5_medmcq-736_out_of_182k",
            #     start_epoch=1,
            #     end_epoch=2,
            #     lora_r=20, 
            #     lora_alpha=40.0,
            #     learning_rate=2e-4,
            #     batch_size=5,
            #     save_steps=150,
            #     logging_steps=1
            # )

            ## Set variables using values received from the request
            # print(f"\n\nFROM THE WORKER: params: {params}\n\n")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: params: {params}")

            config.set_variables(
                training_data_path = params["definition"]["combination_id"] + "/" + os.path.basename(params["training_material"]["training_dataset"]),
                validation_data_path = params["definition"]["combination_id"] + "/" + os.path.basename(params["training_material"]["validation_dataset"]),
                model_dir = "/workspace/meta-llama/Meta-Llama-3-8B-Instruct", # Hard Coded 
                out_path = params["definition"]["combination_id"],
                start_epoch = 1,
                end_epoch = params["general_ft_params"]["epochs"],
                lora_r = params["qlora_params"]["Rank"],
                lora_alpha = params["qlora_params"]["Alpha"],
                learning_rate = params["general_ft_params"]["lr"],
                batch_size = params["general_ft_params"]["batch_size"],
                save_steps = 150,
                logging_steps = 1 # Hard Coded
            )
            # print(f"\n\nFROM THE WORKER: config: {config.get_all_variables()}\n\n")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: config: {config.get_all_variables()}")

            # Convert CSV To JSON
            # print(f"FROM THE WORKER: Converting CSV to JSON...")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Converting CSV to JSON...")
            Data_Prep.llama3_data_preparation(csv_file_path = config.training_data_path, out_file_path = config.out_path)



            # Load dataset
            # print(f"FROM THE WORKER: Loading dataset from {config.training_data_path}")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Loading dataset from {config.training_data_path.replace('.csv', '.json')}")
            dataset = load_dataset_for_training(data_path=config.training_data_path.replace(".csv", ".json"), batch_size=config.batch_size, save_steps=config.save_steps)
            
            
            # Calculate save steps
            if config.save_steps >= int( len(dataset) / config.batch_size ):

                config.save_steps = int( len(dataset) / config.batch_size ) - 5
                # print(f"FROM THE WORKER: Save steps changed to {config.save_steps}")
                LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Save steps changed to {config.save_steps}")



            # Load model and tokenizer
            # print(f"FROM THE WORKER: Loading model and tokenizer from {config.model_dir}")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Loading model and tokenizer from {config.model_dir}")
            model, tokenizer = loading_model_and_tokenizer(model_dir=config.model_dir)
            
            # Training model 
            # print(f"FROM THE WORKER: Starting model training...")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Starting model training...")
            training_model(
                out_path=config.out_path,
                start_epoch=config.start_epoch,
                end_epoch=config.end_epoch,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                model=model,
                tokenizer=tokenizer,
                dataset=dataset
            )
            # print(f"FROM THE WORKER: Model training complete!")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Model training complete!")

            # Upload weights to S3
            LLAMA3.send_log_to_flask(message="\n\nSync with S3")
            sync_to_s3( local_dir = config.out_path, bucket_name = 'queryloop-storage', folder_name = params["definition"]["storage_id"] + "/" + params["training_material"]["training_dataset"].split("/")[1])

            # Send finetune completion request to endpoint
            request_url = f'https://dev.queryloop-ai.com/api/eval_bot/set/finetune/status/{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}'
            payload = {"status": "completed"}
            response = requests.post(request_url, json=payload)
            print(response.json())
            
            LLAMA3.send_log_to_flask(message="\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END OF TRANSMISSION xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        except Exception as e:
            # print(f"FROM THE WORKER: Error setting variables: {e}")
            LLAMA3.send_log_to_flask(message=f"\n\nFROM THE WORKER: Error setting variables: {e}\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END OF TRANSMISSION xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            # Send finetune completion request to endpoint
            request_url = f'https://dev.queryloop-ai.com/api/eval_bot/set/finetune/status/{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}'
            payload = {"status": "failed"}
            response = requests.post(request_url, json=payload)

###################
    @staticmethod
    def send_log_to_flask(message: str = None):
        try:
            url = "http://localhost:5050/log"  # URL of your Flask server's endpoint
            payload = {"message": message}
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            # if response.status_code == 200:
            #     print("Log sent successfully")
            # else:
            #     print(f"Failed to send log: {response.status_code}")
        except Exception as e:
            print(f"Failed to send log: {str(e)}")

            