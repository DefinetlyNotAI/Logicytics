import logging
import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load configurations
config = ConfigParser()
config.read('../../config.ini')

# Get config values
model_name = config.get('VulnScan.generate Settings', 'model_name')
epochs = int(config.get('VulnScan.generate Settings', 'epochs'))
batch_size = int(config.get('VulnScan.generate Settings', 'batch_size'))
learning_rate = float(config.get('VulnScan.generate Settings', 'learning_rate'))
train_data_path = config.get('VulnScan.generate Settings', 'train_data_path')
test_data_path = config.get('VulnScan.generate Settings', 'test_data_path')
save_model_path = config.get('VulnScan.generate Settings', 'save_model_path')

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() and config.getboolean('VulnScan.generate Settings', 'use_cuda') else "cpu")

# Set logging for verbosity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load model tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Sensitive vs Non-sensitive


# Function to load files and create a dataframe
def load_files_from_directory(directory_path):
    files_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(('.txt', '.pdf', '.docx')):  # Assuming text-based files
            try:
                with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
                    content = file.read()
                files_data.append((content, 1 if 'sensitive' in file_name else 0))  # 1: sensitive, 0: non-sensitive
            except Exception as e:
                logger.warning(f"Skipping file {file_name}: {str(e)}")
    return pd.DataFrame(files_data, columns=["text", "label"])


# Load and process data
train_df = load_files_from_directory(train_data_path)
test_df = load_files_from_directory(test_data_path)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_df['text'], train_df['label'], test_size=0.1)


# Tokenization & Encoding
def encode_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')


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
logger.info("Starting training...")
train_results = trainer.train()

# Save the model
logger.info(f"Saving model to {save_model_path}")
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

# Evaluate the model
logger.info("Evaluating model...")
eval_results = trainer.evaluate()

logger.info(f"Evaluation results: {eval_results}")

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
logger.info(f"Training completed. Model saved at {save_model_path}")
