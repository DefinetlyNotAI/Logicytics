import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])

BERT_MODEL_NAME = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

def save_progress_graph(accuracies, filename=f"training_progress.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


# noinspection DuplicatedCode
def train_nn_svm(file_paths, model_type, epochs, save_dir):
    if model_type not in ["svm", "nn"]:
        print(f"Invalid model type: {model_type}. Please choose 'svm' or 'nn'.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    print("Loading data...")
    data, labels = load_data(file_paths)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize model
    print("Initializing model...")
    model = None
    param_grid = None
    if model_type == "svm":
        model = SVC(probability=True, random_state=42)
        param_grid = {
            "C": [1, 10],
            "kernel": ["linear"],
            "gamma": ["scale"],
        }
    elif model_type == "nn":
        model = MLPClassifier(random_state=42, max_iter=200)
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
        }

    # Perform grid search for hyperparameter tuning with parallel processing
    print(f"Training {model_type.upper()} model with hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters: {grid_search.best_params_}")

    # Train with the best model
    accuracies = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        print(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
        accuracies.append(accuracy)

        # Save progress graph
        save_progress_graph(accuracies, filename=os.path.join(save_dir, f"training_progress.png"))

        # Save checkpoint model after every epoch
        if epoch % 1 == 0:
            joblib.dump(best_model, os.path.join(save_dir, f"trained_model_epoch_{epoch + 1}.pkl"))
            print(f"Model checkpoint saved: {os.path.join(save_dir, f'trained_model_epoch_{epoch + 1}.pkl')}")

    # Save final model
    joblib.dump(best_model, os.path.join(save_dir, "trained_model.pkl"))
    print(f"Final model saved as {os.path.join(save_dir, 'trained_model.pkl')}")
    print("Training complete.")


# noinspection DuplicatedCode
def train_rfc(file_paths, save_dir, epochs):
    print("Training model...")

    # Load data
    data, labels = load_data(file_paths)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, predictions)

        print(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
        accuracies.append(accuracy)

        # Save progress plot
        save_progress_graph(accuracies, filename=os.path.join(save_dir, f"training_progress.png"))

        # Save model checkpoint
        joblib.dump(model, os.path.join(save_dir, f"trained_model_epoch_{epoch + 1}.pkl"))
        print(f"Model checkpoint saved: {os.path.join(save_dir, f'trained_model_epoch_{epoch + 1}.pkl')}")

    # Save final model
    joblib.dump(model, os.path.join(save_dir, "trained_model.pkl"))
    print(f"Final model saved as {os.path.join(save_dir, 'trained_model.pkl')}")
    print("Training complete.")


def train_xgboost(X_train, X_test, y_train, y_test, SAVE_DIR):
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


def train_bert(X_train, X_test, y_train, y_test, MAX_LEN, LEARNING_RATE, BATCH_SIZE, EPOCHS, SAVE_DIR):
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
    train_data = TensorDataset(train_encodings.input_ids.to(device), train_encodings.attention_mask.to(device),
                               torch.tensor(y_train).to(device))
    test_data = TensorDataset(test_encodings.input_ids.to(device), test_encodings.attention_mask.to(device),
                              torch.tensor(y_test).to(device))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    print("Training BERT model...")
    model.train_model_v2()
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


def train_lstm(X_train, X_test, y_train, y_test, MAX_FEATURES, LEARNING_RATE, BATCH_SIZE, EPOCHS, SAVE_DIR):
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
    train_data = TensorDataset(torch.tensor(X_train_vec, dtype=torch.long).to(device),
                               torch.tensor(y_train, dtype=torch.float32).to(device))
    test_data = TensorDataset(torch.tensor(X_test_vec, dtype=torch.long).to(device),
                              torch.tensor(y_test, dtype=torch.float32).to(device))

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

def train_model_v3(MODEL_TYPE, DATASET_PATH, SAVE_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_FEATURES, MAX_LEN,
                   TEST_SIZE, ):
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
        train_xgboost(X_train_vec_main, X_test_vec_main, y_train_main, y_test_main, SAVE_DIR)

    elif MODEL_TYPE == "bert":
        train_bert(X_train_main, X_test_main, y_train_main, y_test_main, MAX_LEN, LEARNING_RATE, BATCH_SIZE, EPOCHS,
                   SAVE_DIR)

    elif MODEL_TYPE == "lstm":
        train_lstm(X_train_main, X_test_main, y_train_main, y_test_main, MAX_FEATURES, LEARNING_RATE, BATCH_SIZE,
                   EPOCHS, SAVE_DIR)


def train_model_v2(EPOCHS, DATASET_PATH, MODEL, SAVE_DIR):
    FILE_PATH: list[str] = []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".txt"):
            print(f"Indexed file: {file}")
            FILE_PATH.append(os.path.join(DATASET_PATH, file))
    print(f"Total Indexed file's: {len(FILE_PATH)}")

    if not FILE_PATH:
        print(f"No files found for training. Please ensure '{DATASET_PATH}' contains text files.")
    else:
        train_nn_svm(FILE_PATH, MODEL, EPOCHS, SAVE_DIR)


def train_model_v1(DATASET_PATH, SAVE_DIR, EPOCHS):
    file_path = []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".txt"):
            os.path.join(DATASET_PATH, file)
            print(f"Indexed file: {file}")
            file_path.append(os.path.join(DATASET_PATH, file))
    print(f"Total Indexed file's: {len(file_path)}")

    if not file_path:
        print(f"No files found for training. Please ensure '{DATASET_PATH}' contains text files.")
    else:
        print(f"Found {len(file_path)} files for training.")
        train_rfc(file_path, SAVE_DIR, EPOCHS)


train_model_v1(DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .1Lr4",
               EPOCHS=30)

train_model_v1(DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 50k files with 50KB",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .1Sr5",
               EPOCHS=30)

train_model_v2(EPOCHS=50,
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB",
               MODEL="nn",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .2Ln2")

train_model_v2(EPOCHS=50,
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 50k files with 50KB",
               MODEL="nn",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .2Sn3")

train_model_v2(EPOCHS=50,
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB",
               MODEL="svm",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .3Lv2")

train_model_v2(EPOCHS=50,
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 50k files with 50KB",
               MODEL="svm",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .3Sv3")

train_model_v3(MODEL_TYPE="xgboost",
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .4Lx1", EPOCHS=10, BATCH_SIZE=32,
               LEARNING_RATE=5e-5, MAX_FEATURES=7500, MAX_LEN=128, TEST_SIZE=0.2)

train_model_v3(MODEL_TYPE="lstm",
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .5Ll1", EPOCHS=10, BATCH_SIZE=16,
               LEARNING_RATE=5e-5, MAX_FEATURES=7500, MAX_LEN=128, TEST_SIZE=0.2)

train_model_v3(MODEL_TYPE="bert",
               DATASET_PATH=r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 50k files with 50KB",
               SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .6Sb1", EPOCHS=5, BATCH_SIZE=8,
               LEARNING_RATE=5e-5, MAX_FEATURES=5000, MAX_LEN=128, TEST_SIZE=0.2)
