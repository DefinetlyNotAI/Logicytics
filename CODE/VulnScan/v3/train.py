from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from logicytics import *

if __name__ == "__main__":
    log = Log(
        {"log_level": "Info",
         "filename": "VulnScanTrain.log",
         "colorlog_fmt_parameters":
             "%(log_color)s%(levelname)-8s%(reset)s %(yellow)s%(asctime)s %(blue)s%(message)s",
         }
    )

# Load configurations
config = ConfigParser()
config.read('../../config.ini')

# Get config values
log.info("Loading configurations...")
model_name = config.get('VulnScan.train Settings', 'model_name')
epochs = int(config.get('VulnScan.train Settings', 'epochs'))
batch_size = int(config.get('VulnScan.train Settings', 'batch_size'))
learning_rate = float(config.get('VulnScan.train Settings', 'learning_rate'))
train_data_path = config.get('VulnScan.train Settings', 'train_data_path')
test_data_path = config.get('VulnScan.train Settings', 'test_data_path')
save_model_path = config.get('VulnScan.train Settings', 'save_model_path')

# Validate model name
allowed_models = ["distilbert", "albert", "fasttext", "tabnet", "clip", "t5"]
if not any(model_name.lower().startswith(allowed_model) for allowed_model in allowed_models):
    raise ValueError(f"Invalid model '{model_name}'. Allowed models are: {', '.join(allowed_models)}")

log.info(f"Validated model: {model_name}")


# GPU check
device = torch.device(
    "cuda" if torch.cuda.is_available() and config.getboolean('VulnScan.train Settings', 'use_cuda') else "cpu")

# Load model tokenizer and model
log.info(f"Loading model {model_name}...")
log.info(f"Using device: {device}")
log.info(f"Training for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")
log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
log.info("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Sensitive vs Non-sensitive


# Function to load files and create a dataframe
def load_files_from_directory(directory_path):
    files_data = []
    for file_name in os.listdir(directory_path):
        try:
            with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
                content = file.read()
            files_data.append((content, 1 if 'sensitive' in file_name else 0))  # 1: sensitive, 0: non-sensitive
            log.info(
                f"Loaded file {file_name}: {len(content)} characters, {'sensitive' if 'sensitive' in file_name else 'non-sensitive'}")
        except Exception as e:
            log.warning(f"Skipping file {file_name}: {str(e)}")
    return pd.DataFrame(files_data, columns=["text", "label"])


# Load and process data
log.info("Loading data...")
train_df = load_files_from_directory(train_data_path)
test_df = load_files_from_directory(test_data_path)

# Split data
log.info("Splitting data into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['text'], train_df['label'], test_size=0.1)


# Tokenization & Encoding
def encode_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')


log.info("Tokenizing and encoding texts...")
train_encodings = encode_texts(train_texts)
val_encodings = encode_texts(val_texts)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': torch.tensor(train_labels.tolist())
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': torch.tensor(val_labels.tolist())
})

# Trainer setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=learning_rate,
    report_to=["tensorboard"]  # For tracking progress in TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train model
log.info("Starting training...")
train_results = trainer.train()

# Save the model
log.info(f"Saving model to {save_model_path}")
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

# Evaluate the model
log.info("Evaluating model...")
eval_results = trainer.evaluate()

log.info(f"Evaluation results: {eval_results}")

# Plot training loss and accuracy (Progress visualization)
train_loss = train_results.training_loss
eval_accuracy = eval_results['eval_accuracy']

epochs_range = np.arange(1, epochs + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, eval_accuracy, label='Validation Accuracy')
plt.title("Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()

# Output logs
log.info(f"Training completed. Model saved at {save_model_path}")
