import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------
# CONFIGURATION CONSTANTS
# ---------------------------------------

# General
MODEL_TYPE = "bert"  # Options: "bert", "xgboost", "lstm"
DATASET_PATH = r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB"  # Path to dataset
SAVE_DIR = r"C:\Users\Hp\Desktop\Model Tests\Model Sense .3Lb1"  # Directory to save trained models
EPOCHS = 4  # Number of training epochs
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 5e-5  # Learning rate
MAX_FEATURES = 10000  # For TF-IDF and Embedding
MAX_LEN = 256  # Max token length for BERT and LSTM
TEST_SIZE = 0.3  # Train-test split ratio

# BERT-specific
BERT_MODEL_NAME = "bert-base-uncased"


# ---------------------------------------
# UTILITIES
# ---------------------------------------

def load_data(data_dir):
    """Loads text data and labels from the directory."""
    texts, labels = [], []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
            text = f.read()
            texts.append(text)
            labels.append(1 if "sensitive" in text.lower() else 0)  # Example labeling
            print(f"File {file_name} loaded successfully. Label: {labels[-1]}")
    return texts, np.array(labels)


def evaluate_model(y_true, y_pred):
    """Evaluates the model using standard metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")


# ---------------------------------------
# MODEL TRAINING FUNCTIONS
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_xgboost(X_train, X_test, y_train, y_test):
    """Trains a Gradient Boosting Classifier (XGBoost) with GPU."""
    print("Enabling GPU acceleration...")
    model = xgb.XGBClassifier(tree_method='gpu_hist')  # Enable GPU acceleration
    print("GPU acceleration enabled.")
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate_model(y_test, predictions)
    joblib.dump(model, os.path.join(SAVE_DIR, "xgboost_model.pkl"))
    print("Model saved as xgboost_model.pkl")


def train_bert(X_train, X_test, y_train, y_test):
    """Trains a BERT model with GPU support."""
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare data for training
    print("Preparing data for training...")
    train_data = TensorDataset(train_encodings.input_ids.to(device), train_encodings.attention_mask.to(device), torch.tensor(y_train).to(device))
    test_data = TensorDataset(test_encodings.input_ids.to(device), test_encodings.attention_mask.to(device), torch.tensor(y_test).to(device))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    print("Training BERT model...")
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate model
    print("Evaluating BERT model...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, _ = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    print("BERT model evaluation:")
    evaluate_model(y_test, np.array(predictions))
    model.save_pretrained(os.path.join(SAVE_DIR, "bert_model"))
    print("Model saved as bert_model")


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional, so multiply by 2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        x = self.sigmoid(x)
        return x


def train_lstm(X_train, X_test, y_train, y_test):
    """Trains an LSTM model using PyTorch with GPU support."""
    print("Training LSTM...")
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    print("Preparing LSTM model...")
    vocab_size = X_train_vec.shape[1]
    model = LSTMModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Prepare data for training
    print("Preparing data for training...")
    train_data = TensorDataset(torch.tensor(X_train_vec, dtype=torch.long).to(device), torch.tensor(y_train, dtype=torch.float32).to(device))
    test_data = TensorDataset(torch.tensor(X_test_vec, dtype=torch.long).to(device), torch.tensor(y_test, dtype=torch.float32).to(device))

    print("Training LSTM model...")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        for batch in train_loader:
            print(f"Batch training: {batch}...")
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

    # Evaluate model
    print("Evaluating LSTM model...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            print(f"Batch testing: {batch}...")
            inputs, _ = batch
            outputs = model(inputs)
            predictions.extend((outputs.squeeze() > 0.5).int().cpu().numpy())

    evaluate_model(y_test, np.array(predictions))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_model.pth"))
    print("Model saved as lstm_model.pth")


# ---------------------------------------
# MAIN LOGIC
# ---------------------------------------

if __name__ == "__main__":
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    texts_main, labels_main = load_data(DATASET_PATH)
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(texts_main, labels_main,
                                                                            test_size=TEST_SIZE, random_state=42)

    # Train based on the chosen model
    if MODEL_TYPE == "xgboost":
        vectorizer_main = TfidfVectorizer(max_features=MAX_FEATURES)
        X_train_vec_main = vectorizer_main.fit_transform(X_train_main)
        X_test_vec_main = vectorizer_main.transform(X_test_main)
        train_xgboost(X_train_vec_main, X_test_vec_main, y_train_main, y_test_main)

    elif MODEL_TYPE == "bert":
        train_bert(X_train_main, X_test_main, y_train_main, y_test_main)

    elif MODEL_TYPE == "lstm":
        train_lstm(X_train_main, X_test_main, y_train_main, y_test_main)
