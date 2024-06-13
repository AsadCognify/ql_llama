from qlm.llama3.inference import LLAMA3
from qlm.s3 import sync_from_s3


def generate_prediction_from_llama3(model, tokenizer, query: str, system_prompt: str, max_tokens: int = 512):
    
    # Previous input params
    # epoch: int, bot_name: str, storage_id: str
    ## Download weights to local system
    # sync_from_s3(s3_folder=f"{storage_id}/{bot_name}", local_dir=storage_id)

    ## Load model and tokenizer (Separated generation and model loading)
    # model, tokenizer = LLAMA3.load_model_and_tokenizer(model_dir=storage_id, epoch=epoch, load_base_model=False)

    ## Generate prediction
    prediction = LLAMA3.LLM_call_for_inference(model, tokenizer, question=query, system_prompt=system_prompt, max_tokens=max_tokens)
    # prediction = None
    
    return prediction


def load_model_and_tokenizer_for_llama3(storage_id: str, bot_name: str, epoch: int, load_base_model: bool = False):
    
    ## Download weights to local system
    sync_from_s3(s3_folder=f"{storage_id}/{bot_name}", local_dir=storage_id)

    model, tokenizer = LLAMA3.load_model_and_tokenizer(model_dir=storage_id, epoch=epoch, load_base_model=load_base_model)

    return model, tokenizer