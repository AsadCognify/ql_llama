from qlm.llama3.inference import LLAMA3
from qlm.s3 import sync_from_s3


def generate_prediction_from_llama3(epoch: int, bot_name: str, storage_id: str, query: str, system_prompt: str ): #max_tokens: int = 512
    
    # Download weights to local system
    sync_from_s3(s3_folder=f"{storage_id}/{bot_name}", local_dir=storage_id)

    # Load model and tokenizer
    model, tokenizer = LLAMA3.load_model_and_tokenizer(model_dir=storage_id, epoch=epoch, load_base_model=False)

    # Generate prediction
    prediction = LLAMA3.LLM_call_for_inference(model, tokenizer, question=query, system_prompt=system_prompt, max_tokens=512)

    return prediction