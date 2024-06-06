from queryloop_llm import RUNPOD, Request

if __name__ == "__main__":
    request = {
        "definition": {
            "storage_id": "obscsr-medbpot-002",
            "combination_id": "com-88964-cust-445",
            "email": "a@c.h",
            "action": "finetune",
            "llm": "llama3",
            "get_from": "s3"
        },
        "qlora_params": {
            "Rank": 1,
            "Alpha": 0.5
        },
        "general_ft_params": {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10
        },
        "training_material": {
            "training_dataset": "name",
            "validation_dataset": "name"
        }
    }
    validation = Request.validate(request)
    print(validation)
    finetune = RUNPOD.run_finetuning(id="obscsr-medbpot-002", email="5sVtN@example.com")
    print(finetune)
else:
    print(__name__)