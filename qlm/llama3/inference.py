import os
import torch
from datasets import load_dataset,load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel,PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

class LLAMA3:

    @staticmethod
    def load_model_and_tokenizer(model_dir, base_model_dir = "/workspace/ql_llama/Meta-Llama-3-8B-Instruct", epoch=None, load_base_model=False):

        if load_base_model:
            model = AutoModelForCausalLM.from_pretrained(
                f'{base_model_dir}/model',
                load_in_4bit=True,
                use_auth_token=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(f'{base_model_dir}/tokenizer', trust_remote_code=True)
        else:
            peft_model_id = model_dir+f"/model_{str(epoch)}epoch"
            config = PeftConfig.from_pretrained(peft_model_id)
            model = AutoModelForCausalLM.from_pretrained(f'{base_model_dir}/model', return_dict=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(f'{base_model_dir}/tokenizer')

            #Load the Lora model
            model = PeftModel.from_pretrained(model, peft_model_id)

        model.eval()
        return model,tokenizer

    @staticmethod
    def LLM_call_for_inference(model, tokenizer, question, system_prompt=None, max_tokens=512):
        template = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|> <|start_header_id|>user<|end_header_id|> 
        Question :{question} <|eot_id|> 
        <|start_header_id|>assistant<|end_header_id|>'''
        batch = tokenizer(template, return_tensors='pt').to('cuda')
        #print(len(batch))
        #print(batch)
        #print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=max_tokens)

        result=tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Clear intermediate tensors
        del batch
        del output_tokens
        torch.cuda.empty_cache()
        
        return result


#########################
# VARIABLES
########################

# Path to the directory where the fine-tuned model is saved.
# model_dir = "output/Llama3_finetuning/Llama3-8b_instruct_r=20_a_40_b5_medmcq-736_out_of_182k"
# base_model_dir = "Meta-Llama-3-8B-Instruct"

# True if we want to load base model (not finetuned) for inference Otherwise False
# load_base_model = False

# Number of epochs for which the model is fine-tuned.
# epoch = 1

# Path to the file on which you want to run inference.
# inference_file_path = "/content/drive/MyDrive/RAG_testing/challenge_datasets/medQA/consolidated_dev.csv"

# Path to the file where you want to save the inference results
# ouput_file_path = "/content/drive/MyDrive/RAG_testing/challenge_datasets/medQA/med_devApr19a_137_Llama3_8b_base_med_mcq_gpt4_generated_with_CoT_r=64_alpha=16_lr=2e-4_b20e1_CoT_test.csv"

# To run an inference
# system_prompt = "You are presented with the following multiple choice question. Think step by step and then select the best answer. just return the correct option with its number"
# query = "Preferred drug for the treatment of a 48 year old man with uncomplicated grade 2 hypeension without any associated co-morbidity is (1) Chlohalidone (2) Triamterene (3) Spironolactone (4) Furosemide"


######################
# MODEL
######################

# inference_model,inference_tokenizer = load_model_and_tokenizer(base_model_dir=base_model_dir, model_dir=model_dir, epoch=epoch, load_base_model=load_base_model)
# result = LLM_call_for_inference(inference_model, inference_tokenizer, system_prompt, query)
# print(f"result:: {result}")