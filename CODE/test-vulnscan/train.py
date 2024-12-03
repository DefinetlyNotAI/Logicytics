import logging
import os
from os import mkdir

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
                        logging.FileHandler("Training.log"),
                        logging.StreamHandler()
                    ])

BERT_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


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
            logging.info(f"File {file_name} loaded successfully. Label: {labels[-1]}")
    return texts, np.array(labels)


def evaluate_model(y_true, y_pred):
    """Evaluates the model using standard metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    roc_auc = roc_auc_score(y_true, y_pred) if len(set(y_pred)) > 1 else float('nan')
    logging.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    return accuracy, precision, recall, f1, roc_auc


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
def train_nn_svm(MODEL, EPOCHS, SAVE_DIR, MAX_FEATURES, TEST_SIZE, MAX_ITER, RANDOM_STATE):
    if MODEL not in ["svm", "nn"]:
        logging.error(f"Invalid model type: {MODEL}. Please choose 'svm' or 'nn'.")
        return

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load data
    logging.info("Loading data...")
    data, labels = DATA
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # Initialize model
    logging.info("Initializing model...")
    if MODEL == "svm":
        model = SVC(probability=True, random_state=RANDOM_STATE)
        param_grid = {
            "C": [1, 10],
            "kernel": ["linear"],
            "gamma": ["scale"],
        }
    elif MODEL == "nn":
        model = MLPClassifier(random_state=RANDOM_STATE, max_iter=MAX_ITER)
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
        }
    else:
        logging.error(f"Invalid model type: {MODEL}. Please choose 'svm' or 'nn'.")
        return

    # Perform grid search for hyperparameter tuning with parallel processing
    logging.info(f"Training {MODEL.upper()} model with hyperparameter tuning...")
    try:
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1)
        grid_search.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Error occurred during grid search: {e}")
        logging.info("Training with CV=2...")
        grid_search = GridSearchCV(model, param_grid, cv=2, scoring="accuracy", verbose=1)
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best Model Parameters: {grid_search.best_params_}")

    # Train with the best model
    accuracies = []
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) if len(set(y_test)) > 1 else float(
            'nan')
        logging.info(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
        accuracies.append(accuracy)

        # Save progress graph
        save_progress_graph(accuracies, filename=os.path.join(SAVE_DIR, f"training_progress.png"))

        # Save checkpoint model after every epoch
        if epoch % 1 == 0:
            joblib.dump(best_model, os.path.join(SAVE_DIR, f"trained_model_epoch_{epoch + 1}.pkl"))
            logging.info(f"Model checkpoint saved: {os.path.join(SAVE_DIR, f'trained_model_epoch_{epoch + 1}.pkl')}")

    # Save final model
    joblib.dump(best_model, os.path.join(SAVE_DIR, "trained_model.pkl"))
    logging.info(f"Final model saved as {os.path.join(SAVE_DIR, 'trained_model.pkl')}")
    logging.info("Training complete.")


# noinspection DuplicatedCode
def train_rfc(SAVE_DIR, EPOCHS, TEST_SIZE, N_ESTIMATORS, RANDOM_STATE):
    logging.info("Training model...")

    # Load data
    data, labels = DATA

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(data).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Initialize model
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    accuracies = []

    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, predictions) if len(set(y_test)) > 1 else float('nan')
        logging.info(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
        accuracies.append(accuracy)

        # Save progress plot
        if not os.path.exists(SAVE_DIR):
            mkdir(SAVE_DIR)
        save_progress_graph(accuracies, filename=os.path.join(SAVE_DIR, f"training_progress.png"))

        # Save model checkpoint
        joblib.dump(model, os.path.join(SAVE_DIR, f"trained_model_epoch_{epoch + 1}.pkl"))
        logging.info(f"Model checkpoint saved: {os.path.join(SAVE_DIR, f'trained_model_epoch_{epoch + 1}.pkl')}")

    # Save final model
    joblib.dump(model, os.path.join(SAVE_DIR, "trained_model.pkl"))
    logging.info(f"Final model saved as {os.path.join(SAVE_DIR, 'trained_model.pkl')}")
    logging.info("Training complete.")


def train_xgboost(X_train, X_test, y_train, y_test, SAVE_DIR):
    """Trains a Gradient Boosting Classifier (XGBoost) with GPU."""
    logging.info("Enabling GPU acceleration...")
    model = xgb.XGBClassifier(tree_method='hist', device=DEVICE)  # Enable GPU acceleration
    logging.info("GPU acceleration enabled.")
    logging.info("Training XGBoost...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logging.info("XGBoost evaluation commencing...")
    evaluate_model(y_test, predictions)
    joblib.dump(model, os.path.join(SAVE_DIR, "xgboost_model.pkl"))
    logging.info("Model saved as xgboost_model.pkl")


def train_bert(X_train, X_test, y_train, y_test, MAX_LEN, LEARNING_RATE, BATCH_SIZE, EPOCHS, SAVE_DIR, MODEL_PATH):
    """Trains a BERT model with GPU support."""
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    logging.info("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare data for training
    logging.info("Preparing data for training...")
    train_data = TensorDataset(train_encodings.input_ids.to(DEVICE), train_encodings.attention_mask.to(DEVICE),
                               torch.tensor(y_train).to(DEVICE))
    test_data = TensorDataset(test_encodings.input_ids.to(DEVICE), test_encodings.attention_mask.to(DEVICE),
                              torch.tensor(y_test).to(DEVICE))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    logging.info("Training BERT model...")
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate model
    logging.info("Evaluating BERT model...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, _ = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    logging.info("BERT model evaluating..")
    evaluate_model(y_test, np.array(predictions))
    model.save_pretrained(os.path.join(SAVE_DIR, "bert_model"))
    logging.info("Model saved as bert_model")


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
    logging.info("Training LSTM...")
    logging.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    logging.info("Preparing LSTM model...")
    vocab_size = X_train_vec.shape[1]
    model = LSTMModel(vocab_size=vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Prepare data for training
    logging.info("Preparing data for training...")
    train_data = TensorDataset(torch.tensor(X_train_vec, dtype=torch.long).to(DEVICE),
                               torch.tensor(y_train, dtype=torch.float32).to(DEVICE))
    test_data = TensorDataset(torch.tensor(X_test_vec, dtype=torch.long).to(DEVICE),
                              torch.tensor(y_test, dtype=torch.float32).to(DEVICE))

    logging.info("Training LSTM model...")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        for batch in train_loader:
            logging.info(f"Batch training: {batch}...")
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

    # Evaluate model
    logging.info("Evaluating LSTM model...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            logging.info(f"Batch testing: {batch}...")
            inputs, _ = batch
            outputs = model(inputs)
            predictions.extend((outputs.squeeze(dim=-1) > 0.5).int().cpu().numpy())

    evaluate_model(y_test, np.array(predictions))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_model.pth"))
    logging.info("Model saved as lstm_model.pth")


# ---------------------------------------
# MAIN LOGIC
# ---------------------------------------

def train_model_v3(MODEL_TYPE, SAVE_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_FEATURES, MAX_LEN,
                   TEST_SIZE, RANDOM_STATE, MODEL_PATH_BERT=None):
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    texts_main, labels_main = DATA
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(texts_main, labels_main,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)

    # Train based on the chosen model
    if MODEL_TYPE == "xgboost":
        vectorizer_main = TfidfVectorizer(max_features=MAX_FEATURES)
        X_train_vec_main = vectorizer_main.fit_transform(X_train_main)
        X_test_vec_main = vectorizer_main.transform(X_test_main)
        train_xgboost(X_train_vec_main, X_test_vec_main, y_train_main, y_test_main, SAVE_DIR)

    elif MODEL_TYPE == "bert":
        if MODEL_PATH_BERT is None:
            logging.error("Please provide a valid BERT model path.")
            return
        train_bert(X_train_main, X_test_main, y_train_main, y_test_main, MAX_LEN, LEARNING_RATE, BATCH_SIZE, EPOCHS,
                   SAVE_DIR, MODEL_PATH_BERT)

    elif MODEL_TYPE == "lstm":
        train_lstm(X_train_main, X_test_main, y_train_main, y_test_main, MAX_FEATURES, LEARNING_RATE, BATCH_SIZE,
                   EPOCHS, SAVE_DIR)


if __name__ == "__main__":
    DATA = load_data(r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB")

    train_rfc(SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .1Lr4", EPOCHS=30, TEST_SIZE=0.2,
              N_ESTIMATORS=100, RANDOM_STATE=42)

    train_rfc(SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .1Sr5", EPOCHS=30, TEST_SIZE=0.2,
              N_ESTIMATORS=100, RANDOM_STATE=42)

    train_nn_svm(EPOCHS=50,
                 MODEL="nn", SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .2Ln2", MAX_FEATURES=5000,
                 TEST_SIZE=0.2, MAX_ITER=5000, RANDOM_STATE=42)

    train_nn_svm(EPOCHS=50,
                 MODEL="nn", SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .2Sn3", MAX_FEATURES=5000,
                 TEST_SIZE=0.2, MAX_ITER=5000, RANDOM_STATE=42)

    train_nn_svm(EPOCHS=50,
                 MODEL="svm", SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .3Lv2", MAX_FEATURES=5000,
                 TEST_SIZE=0.2, MAX_ITER=5000, RANDOM_STATE=42)

    train_nn_svm(EPOCHS=50,
                 MODEL="svm", SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .3Sv3", MAX_FEATURES=5000,
                 TEST_SIZE=0.2, MAX_ITER=5000, RANDOM_STATE=42)

    train_model_v3(MODEL_TYPE="xgboost",
                   SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .4Lx1", EPOCHS=10, BATCH_SIZE=32,
                   LEARNING_RATE=5e-5, MAX_FEATURES=7500, MAX_LEN=128, TEST_SIZE=0.2, RANDOM_STATE=42)

    train_model_v3(MODEL_TYPE="lstm",
                   SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .5Ll1", EPOCHS=10, BATCH_SIZE=16,
                   LEARNING_RATE=5e-5, MAX_FEATURES=7500, MAX_LEN=128, TEST_SIZE=0.2, RANDOM_STATE=42)

    train_model_v3(MODEL_TYPE="bert",
                   SAVE_DIR=r"C:\Users\Hp\Desktop\Model Tests\Model Sense .6Sb1", EPOCHS=5, BATCH_SIZE=8,
                   LEARNING_RATE=5e-5, MAX_FEATURES=5000, MAX_LEN=128, TEST_SIZE=0.2, RANDOM_STATE=42,
                   MODEL_PATH_BERT="Extra/bert-base-uncased-model")
