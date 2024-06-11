import time
from typing import Dict, Any
from qlm.llama3.llama3_finetuning import loading_model_and_tokenizer, training_model, load_dataset_for_training
from qlm.llama3.finetuning_variables import LLAMA3TrainingConfig

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
            # Define variables
            ## Initalize config
            config = LLAMA3TrainingConfig()
            print(f"\n\nFROM THE WORKER: Creating instance of LLAMA3TrainingConfig. {config}\n\n")
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
            print(f"\n\nFROM THE WORKER: params: {params}\n\n")

            config.set_variables(
                training_data_path = params["definition"]["combination_id"] + "/training_dataset.json",
                validation_data_path = params["definition"]["combination_id"] + "/validation_dataset.json",
                model_dir = "/workspace/ql_llama/Meta-Llama-3-8B-Instruct", # Hard Coded 
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
            print(f"\n\nFROM THE WORKER: config: {config.get_all_variables()}\n\n")

            # Load dataset
            print(f"FROM THE WORKER: Loading dataset from {config.training_data_path}")
            dataset = load_dataset_for_training(data_path=config.training_data_path, batch_size=config.batch_size, save_steps=config.save_steps)
            
            
            # Calculate save steps
            if config.save_steps >= int( len(dataset) / config.batch_size ):

                config.save_steps = int( len(dataset) / config.batch_size ) - 5
                print(f"FROM THE WORKER: Save steps changed to {config.save_steps}")



            # Load model and tokenizer
            model, tokenizer = loading_model_and_tokenizer(model_dir=config.model_dir)
            
            # Training model 
            print(f"FROM THE WORKER: Starting model training...")
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
            print(f"FROM THE WORKER: Model training complete!")

        except Exception as e:
            print(f"FROM THE WORKER: Error setting variables: {e}")