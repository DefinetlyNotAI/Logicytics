# TODO Add more verbose logging, and add type hints + docstrings
import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader

# Set up logging
from logicytics import Log

logger = Log(
    {"log_level": "Info",
     "filename": "VulnScanTrain.log",
     "colorlog_fmt_parameters":
         "%(log_color)s%(levelname)-8s%(reset)s %(yellow)s%(asctime)s %(blue)s%(message)s",
     }
)


# Dataset Class for PyTorch models
class SensitiveDataDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            text = self.tokenizer(text)
        return torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Train Model Function
def train_model(
        model_name,
        epochs,
        batch_size,
        learning_rate,
        train_data_path,
        save_model_path,
        use_cuda=False,
):
    # Load Data
    logger.info(f"Loading data from {train_data_path}")
    texts, labels = [], []
    for filename in os.listdir(train_data_path):
        with open(os.path.join(train_data_path, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(1 if '-sensitive' in filename else 0)
        logger.info(f"Loaded data from {filename}")

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vectorizer Setup
    vectorizer = None
    if model_name in ['Tfidf_LogReg', 'CountVectorizer_LogReg']:
        vectorizer_type = 'Tfidf' if 'Tfidf' in model_name else 'Count'
        logger.info(f"Using Vectorizer {vectorizer_type}")
        vectorizer = TfidfVectorizer(max_features=10000,
                                     ngram_range=(1, 2)) if vectorizer_type == 'Tfidf' else CountVectorizer(
            max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_val = vectorizer.transform(X_val).toarray()

    # Model Selection and Training
    if model_name == 'NeuralNetwork':
        # Vectorize the text data
        logger.info("Vectorizing text data for Neural Network")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_val = vectorizer.transform(X_val).toarray()

        model = nn.Sequential(nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Linear(128, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model.to(device)

        # DataLoader
        train_dataset = SensitiveDataDataset(X_train, y_train)
        val_dataset = SensitiveDataDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        acc = []
        loss_plot = []

        for epoch in range(epochs):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc.append(correct / total)
            loss_plot.append(epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {(correct / total):.4f}")

        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            model.eval()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    else:
        # Ensure text is vectorized for non-NN models
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            X_train = vectorizer.fit_transform(X_train).toarray()
            X_val = vectorizer.transform(X_val).toarray()

        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=epochs)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(n_estimators=100)
        elif model_name == 'GBM':
            model = GradientBoostingClassifier(n_estimators=100)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(eval_metric='logloss')
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier()
        elif model_name == 'NaiveBayes':
            model = MultinomialNB()
        elif model_name == 'LogReg':
            model = LogisticRegression(max_iter=epochs)
        else:
            logger.error(f"Invalid model name: {model_name}")
            return

        # Train Traditional Model
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        logger.info(f"Validation Accuracy: {acc:.4f}")
        logger.info(classification_report(y_val, preds))

        # Store loss and accuracy for each epoch
        loss_plot = []
        acc_plot = []

        # Train Traditional Model
        for epoch in range(epochs):
            model.fit(X_train, y_train)  # You can also track partial fit for large datasets

            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            acc_plot.append(acc)

            logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {acc:.4f}")
            logger.info(classification_report(y_val, preds, zero_division=0))

            # Optional: Track loss (e.g., log loss for XGBoost)
            # For models without loss, you can track the error rate or use other metrics
            if hasattr(model, 'predict_proba'):
                loss = -model.score(X_val, y_val)  # In case model supports probability prediction
            else:
                loss = 1 - acc  # Simple approximation for loss based on accuracy
            loss_plot.append(loss)

        # Plot the accuracy and loss over epochs
        plt.figure(figsize=(12, 6))

        # Accuracy Plot
        plt.plot(list(range(1, epochs + 1)), loss_plot, label="Accuracy")
        plt.title(f'{model_name} - Validation Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(save_model_path), f"Model Accuracy Over Epochs - {model_name}.png"))
        plt.show()

    # Save Model
    if save_model_path:
        logger.info(f"Saving model to {save_model_path}.pth")
        torch.save(model, save_model_path + ".pth")

    # Visuals (NN)
    if model_name == 'NeuralNetwork':
        plt.plot(list(range(1, epochs + 1)), acc, label="Training Accuracy")
        plt.title("Model Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(save_model_path), f"Model Accuracy Over Epochs - {model_name}.png"))
        plt.show()

        plt.plot(list(range(1, epochs + 1)), loss_plot, label="Training Loss")
        plt.title("Model Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(save_model_path), f"Model Loss Over Epochs - {model_name}.png"))
        plt.show()


# Config file reading
config = ConfigParser()
config.read('../../config.ini')
if config.getboolean('VulnScan.train Settings', 'use_1_model_only?'):
    try:
        train_model(model_name=config.get('VulnScan.train Settings', 'model_name'),
                    epochs=int(config.get('VulnScan.train Settings', 'epochs')),
                    batch_size=int(config.get('VulnScan.train Settings', 'batch_size')),
                    learning_rate=float(config.get('VulnScan.train Settings', 'learning_rate')),
                    train_data_path=config.get('VulnScan.train Settings', 'train_data_path'),
                    save_model_path=config.get('VulnScan.train Settings', 'save_model_path'),
                    use_cuda=config.getboolean('VulnScan.train Settings', 'use_cuda'))
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error in training model: {e}")
        exit(1)
    except AttributeError as e:
        logger.error(f"Attribute Error in training model: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        exit(1)
else:
    for model_main in ["NeuralNetwork", "LogReg",
                       "RandomForest", "ExtraTrees", "GBM",
                       "XGBoost", "DecisionTree", "NaiveBayes"]:

        code_names = {
            "DecisionTree": "dt",
            "ExtraTrees": "et",
            "GBM": "g",
            "NeuralNetwork": "n",
            "NaiveBayes": "nb",
            "RandomForest": "r",
            "LogReg": "lr",
            "XGBoost": "x"
        }

        if model_main in code_names.keys():
            model_path_save = f"{config.get('VulnScan.train Settings', 'save_model_path')} 3{code_names[model_main]}1"

        logger.info(f"Save path: {model_path_save}")

        try:
            train_model(model_name=model_main,
                        epochs=int(config.get('VulnScan.train Settings', 'epochs')),
                        batch_size=int(config.get('VulnScan.train Settings', 'batch_size')),
                        learning_rate=float(config.get('VulnScan.train Settings', 'learning_rate')),
                        train_data_path=config.get('VulnScan.train Settings', 'train_data_path'),
                        save_model_path=model_path_save,
                        use_cuda=config.getboolean('VulnScan.train Settings', 'use_cuda'))
        except FileNotFoundError as e:
            logger.error(f"File Not Found Error in training model {model_main}: {e}")
            exit(1)
        except AttributeError as e:
            logger.error(f"Attribute Error in training model {model_main}: {e}")
            exit(1)
        except Exception as e:
            logger.error(f"Error in training model {model_main}: {e}")
            exit(1)
