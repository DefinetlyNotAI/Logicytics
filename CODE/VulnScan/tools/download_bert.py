from transformers import BertForSequenceClassification

# Define the model name
BERT_MODEL_NAME = "bert-base-uncased"

# Download and save the model
print(f"Downloading and saving '{BERT_MODEL_NAME}' model...")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
model.save_pretrained("bert-base-uncased-model")
print(f"Model saved to 'bert-base-uncased-model'")
