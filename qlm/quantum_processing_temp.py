import gc
import os
import time
import torch
import shutil
import requests
import pandas as pd
from typing import Dict, Any
from qlm.s3 import sync_to_s3
from qlm.utils.logger import logger
from qlm.llama3.data_prep import Data_Prep
from qlm.llama3.finetuning_variables import LLAMA3TrainingConfig
from qlm.llama3.llama3_finetuning import loading_model_and_tokenizer, training_model, load_dataset_for_training, load_dataset_for_validation

class LLAMA3:
    def __init__(self) -> None:
        pass

    @classmethod
    def _update_mongo(cls, status: str, bot_endpoint: str, loss: float = None):
        logger.debug(f"Updating mongo status to: {status}")
        request_url = f'https://stage.queryloop-ai.com/api/eval_bot/update/combination/finetune/{bot_endpoint}'
        logger.debug(f"request_url: {request_url}")
        payload = {"status": status} #running #failed #compeleted
        response = requests.post(request_url, json=payload)
        logger.debug(f"response: {response.json()}")


    @classmethod
    def _convert_dataset_from_csv_to_json(cls, dataset_path: str, out_path: str):
        logger.debug(f"Converting training CSV to JSON...")
        Data_Prep.llama3_data_preparation(csv_file_path = dataset_path, out_file_path = out_path)

    @classmethod
    def _modify_save_steps(len_training_dataset: int, batch_size, current_save_steps: int):
        # Calculate save steps
        if current_save_steps >= int( len_training_dataset / batch_size ):
            current_save_steps = int( len_training_dataset / batch_size ) - 5
        logger.info(f"Save steps changed to {current_save_steps}")

        return current_save_steps

    @staticmethod
    def finetune(params: Dict[str, Any], ft: str = None):

        try:
            # Update mongo to running status
            LLAMA3._update_mongo(status='running', bot_endpoint=f'{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}')

            # Create config dict
            logger.info(f"parms: {params}")
            config = {
                "training_data_path" : f"/home/" + params["definition"]["combination_id"] + "/" + os.path.basename(params["training_material"]["training_dataset"]),
                "validation_data_path" : f"/home/" + params["definition"]["combination_id"] + "/" + os.path.basename(params["training_material"]["validation_dataset"]),
                "model_dir" : "/workspace/meta-llama/Meta-Llama-3-8B-Instruct", # Hard Coded 
                "out_path" : f"/home/" + params["definition"]["combination_id"], #/bot/<combination_id>/<model_epoch>
                "start_epoch" : 1,
                "end_epoch" : params["general_ft_params"]["epochs"],
                "lora_r" : params["qlora_params"]["Rank"],
                "lora_alpha" : params["qlora_params"]["Alpha"],
                "learning_rate" : params["general_ft_params"]["lr"],
                "batch_size" : params["general_ft_params"]["batch_size"],
                "save_steps" : 2, # For testing
                "logging_steps" : 1 # Hard Coded
            }
            logger.info(f"config: {config}")


            # Convert datasets from CSV To JSON
            logger.debug(f"Converting training CSV to JSON...")
            Data_Prep.llama3_data_preparation(csv_file_path = config["training_data_path"], out_file_path = config["out_path"])

            # Validation file
            logger.debug(f"Converting validation CSV to JSON...")
            Data_Prep.llama3_data_preparation(csv_file_path = config["validation_data_path"], out_file_path = config["out_path"])

            # Load datasets for training and validation
            logger.debug(f"Loading dataset from {config['training_data_path'].replace('.csv', '.json')}")
            training_dataset = load_dataset_for_training(
                data_path=config["training_data_path"].replace(".csv", ".json")
            )
            
            logger.debug(f"Loading dataset from {config['validation_data_path'].replace('.csv', '.json')}")
            validation_dataset = load_dataset_for_validation(
                data_path=config["validation_data_path"].replace(".csv", ".json")
            )

            logger.info(f"Loading datasets complete!")

            # Modify save steps
            logger.debug(f"Evaluating training dataset\nLength of training dataset: {len(training_dataset)}\n{training_dataset}")
            
            # modify batch size if necessary
            config["batch_size"] = min(config["batch_size"], len(training_dataset))
            logger.info(f"batch size: {config['batch_size']}")

            config['save_steps'] = min(1, config['batch_size'])
            logger.info(f"save steps: {config['save_steps']}")

            # laod model and tokenizer
            logger.debug(f"Loading model and tokenizer from {config['model_dir']}")
            model, tokenizer = loading_model_and_tokenizer(model_dir=config["model_dir"])

            # start training
            logger.info(f"Starting model training...")
            training_model(
                out_path=config["out_path"],
                start_epoch=config["start_epoch"],
                end_epoch=config["end_epoch"],
                lora_r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                logging_steps=config["logging_steps"],
                save_steps=config["save_steps"],
                model=model,
                tokenizer=tokenizer,
                dataset=training_dataset,
                val_dataset=validation_dataset
            )
            logger.debug(f"Model training complete!")

            # Sync with S3 (Skipped)
            logger.warning(f"Sync with S3")
            sync_to_s3( local_dir = config["out_path"], bucket_name = 'queryloop-storage', folder_name = params["definition"]["storage_id"] + "/" + params["definition"]["bot_name"] + "/" + params["definition"]["combination_id"])

            # return eval loss if bayesian fintuning
            if ft == 'bayesian':
                path_to_loss_file = config["out_path"]+f'/loss_{config["end_epoch"]}epoch.csv'
                eval_loss_df = pd.read_csv(path_to_loss_file)
                cleaned_losses = eval_loss_df['eval_loss'].dropna()

                logger.info(f"Cleaning up folder: {config['out_path']}")
                # os.removedirs(config["out_path"])
                shutil.rmtree(config["out_path"])

                # Update mongo status
                LLAMA3._update_mongo(status='complete', loss=cleaned_losses.iloc[-1], bot_endpoint=f'{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}')

                # Removing loaded model from GPU
                logger.info("Clearing up GPU memory")
                gc.collect()
                torch.cuda.empty_cache()
                del model
                del tokenizer
    
                logger.info(f"\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END OF TRANSMISSION xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                
                return cleaned_losses.iloc[-1]

            # Retrieve loss for grid search
            path_to_loss_file = config["out_path"]+f'/loss_{config["end_epoch"]}epoch.csv'
            eval_loss_df = pd.read_csv(path_to_loss_file)
            cleaned_losses = eval_loss_df['eval_loss'].dropna()


            # Update mongo status
            LLAMA3._update_mongo(status='complete', loss=cleaned_losses.iloc[-1], bot_endpoint=f'{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}')

            logger.info(f"Cleaning up folder: {config['out_path']}")
            # os.removedirs(config["out_path"])
            shutil.rmtree(config["out_path"])
            logger.info(f"\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END OF TRANSMISSION xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
        except Exception as e:
            logger.error(f"Error during finetuning: {e}\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END OF TRANSMISSION xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", exc_info=True)

            # Send finetune completion request to endpoint
            LLAMA3._update_mongo(status='failed', bot_endpoint=f'{params["definition"]["bot_id"]}/{params["definition"]["combination_id"]}')


