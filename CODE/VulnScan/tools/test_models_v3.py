from transformers import AutoModel, AutoTokenizer

# List of some available models
models = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "gpt2",
    "t5-small",
    "xlm-roberta-base"
]

for model_name in models:
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model {model_name} is available.")
    except Exception as e:
        print(f"Model {model_name} is not available: {e}")
