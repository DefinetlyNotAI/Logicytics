from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, \
    AlbertForSequenceClassification, T5ForConditionalGeneration

# Define model names and paths
MODEL_DETAILS = {
    "bert-base-uncased": ("transformers", BertForSequenceClassification),
    "distilbert-base-uncased": ("transformers", DistilBertForSequenceClassification),
    "albert-base-v2": ("transformers", AlbertForSequenceClassification),
    "t5-small": ("transformers", T5ForConditionalGeneration),
}

# Download and save models
for model_name, (library, model_class) in MODEL_DETAILS.items():
    print(f"Downloading and saving '{model_name}' using '{library}'...")
    if library == "transformers":
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(f"../{model_name.split('-')[0]}")
    elif library == "tabnet":
        print(f"TabNet model setup for '{model_name}' is placeholder-only (custom model not supported yet).")
    print(f"Model '{model_name}' saved successfully!")

print("All models downloaded and saved.")
