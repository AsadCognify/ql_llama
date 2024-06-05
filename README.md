# ql_llama

## Steps to follow through
1. Clone the repo using the latest tag release
2. Create a virtual encironment using `python -m venv venv` or other command
3. Install huggingface cli using `pip install huggingface-hub`
4. Download LlaMA 3 8B Instruct using the command `huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*" --local-dir ./Meta-Llama-3-8B-Instruct`
5. Run Finetuning or Inference
6. Additional data prep file available for a specific dataset
