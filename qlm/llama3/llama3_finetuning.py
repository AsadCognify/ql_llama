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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd
from .finetuning_variables import LLAMA3TrainingConfig

def loading_model_and_tokenizer(model_dir):
  model = AutoModelForCausalLM.from_pretrained(
      f'{model_dir}/model',
      load_in_4bit=True,
      device_map="auto",
      trust_remote_code=True,
  )
  model.config.use_cache = False
  model.config.pretraining_tp = 1

  # Load LLaMA tokenizer
  tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/tokenizer', trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  return model,tokenizer


def training_model(out_path,start_epoch,end_epoch,lora_r,lora_alpha,learning_rate,batch_size,logging_steps,save_steps,model,tokenizer,dataset):


  ################################################################################
  # QLoRA parameters
  ################################################################################


  lora_r = lora_r

  # Alpha parameterter for LoRA scaling
  lora_alpha = lora_alpha

  # Dropout probability for LoRA layers
  lora_dropout = 0.1

  ################################################################################
  # bitsandbytes parameters
  ################################################################################

  # Activate 4-bit precision base model loading
  use_4bit = True

  # Compute dtype for 4-bit base models
  bnb_4bit_compute_dtype = "float16"

  # Quantization type (fp4 or nf4)
  bnb_4bit_quant_type = "nf4"

  # Activate nested quantization for 4-bit base models (double quantization)
  use_nested_quant = False

  ################################################################################
  # TrainingArguments parameters
  ################################################################################

  # Output directory where the model predictions and checkpoints will be stored
  output_dir = out_path+ "/checkpoint"

  # Check if the directory exists
  if not os.path.exists(output_dir):
      # Create the directory
      os.makedirs(output_dir)


  # Number of training epochs
  num_train_epochs = start_epoch

  # Enable fp16/bf16 training (set bf16 to True with an A100)
  fp16 = False
  bf16 = False

  # Batch size per GPU for training
  per_device_train_batch_size = batch_size

  # Batch size per GPU for evaluation
  per_device_eval_batch_size = batch_size

  # Number of update steps to accumulate the gradients for
  gradient_accumulation_steps = 1

  # Enable gradient checkpointing
  gradient_checkpointing = True

  # Maximum gradient normal (gradient clipping)
  max_grad_norm = 0.3

  # Initial learning rate (AdamW optimizer)
  learning_rate = learning_rate

  # Weight decay to apply to all layers except bias/LayerNorm weights
  weight_decay = 0.001

  # Optimizer to use
  optim = "paged_adamw_32bit"

  # Learning rate schedule
  lr_scheduler_type = "cosine"

  # Number of training steps (overrides num_train_epochs)
  max_steps = -1

  # Ratio of steps for a linear warmup (from 0 to learning rate)
  warmup_ratio = 0.03

  # Group sequences into batches with same length
  # Saves memory and speeds up training considerably
  group_by_length = True

  # Save checkpoint every X updates steps
  save_steps = save_steps

  # Log every X updates steps
  logging_steps = logging_steps

  ################################################################################
  # SFT parameters
  ################################################################################

  # Maximum sequence length to use
  max_seq_length = None

  # Pack multiple short examples in the same input sequence to increase efficiency
  packing = False

  # Load the entire model on the GPU 0
  device_map = {"": 0}




  compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=use_4bit,
      bnb_4bit_quant_type=bnb_4bit_quant_type,
      bnb_4bit_compute_dtype=compute_dtype,
      bnb_4bit_use_double_quant=use_nested_quant,
  )

  # Check GPU compatibility with bfloat16
  if compute_dtype == torch.float16 and use_4bit:
      major, _ = torch.cuda.get_device_capability()
      if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


  # Load LoRA configuration
  peft_config = LoraConfig(
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      r=lora_r,
      bias="none",
      task_type="CAUSAL_LM",
  )

  # Set training parameters
  report_to = None

  while start_epoch <=end_epoch:
    if start_epoch ==1 :
      resume_from_checkpoint=False
    else:
      resume_from_checkpoint=True


    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=start_epoch,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    peft_model_id=out_path+f'/model_{start_epoch}epoch'
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    loss_df=pd.DataFrame(trainer.state.log_history)
    loss_df.to_csv(out_path+f'/loss_{start_epoch}epoch.csv')
    start_epoch=start_epoch+1


def load_dataset_for_training(data_path: str, batch_size: int, save_steps: int):
    ###################
    #     Dataset     #
    ###################
    dataset = load_dataset("json", data_files=data_path, field="data")
    dataset = dataset['train']

    return dataset


def train(model_dir: str):
    ###########################
    #    Train
    ###########################
    model,tokenizer = loading_model_and_tokenizer(model_dir)
    training_model(out_path,start_epoch,end_epoch,lora_r,lora_alpha,learning_rate,batch_size,logging_steps,save_steps,model,tokenizer,dataset)