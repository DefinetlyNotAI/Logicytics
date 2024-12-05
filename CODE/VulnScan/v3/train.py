import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from configparser import ConfigParser

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
        model_name=None,
        epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        train_data_path=None,
        save_model_path=None,
        use_cuda=True,
):
    # Load Data
    logger.info(f"Loading data from {train_data_path}")
    texts, labels = [], []
    for filename in os.listdir(train_data_path):
        with open(os.path.join(train_data_path, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(1 if '-sensitive' in filename else 0)

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vectorizer Setup
    if model_name in ['Tfidf_LogReg', 'CountVectorizer_LogReg']:
        vectorizer_type = 'Tfidf' if 'Tfidf' in model_name else 'Count'
        logger.info(f"Using Vectorizer {vectorizer_type}")
        vectorizer = TfidfVectorizer(max_features=10000,
                                     ngram_range=(1, 2)) if vectorizer_type == 'Tfidf' else CountVectorizer(
            max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_val = vectorizer.transform(X_val).toarray()

    # Model Selection
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
                preds = preds.to(torch.long)
                labels = labels.to(torch.long)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = correct / total
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Check the shapes before and after squeezing
                logger.info(
                    f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")

                # Get predicted classes
                _, preds = torch.max(outputs, 1)
                logger.info(f"Preds shape: {preds.shape}, Labels shape: {labels.shape}")

                # Ensure correct shape for preds and labels
                labels = labels.to(torch.long)  # Ensure labels are in the correct type
                preds = preds.to(torch.long)  # Ensure preds are in the correct type

                # Check if the shapes match before comparison
                logger.info(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")

                if preds.shape != labels.shape:
                    logger.error(f"Shape mismatch: preds {preds.shape}, labels {labels.shape}")

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    else:
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=epochs)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(n_estimators=100)
        elif model_name == 'GBM':
            model = GradientBoostingClassifier(n_estimators=100)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier()
        elif model_name == 'NaiveBayes':
            model = MultinomialNB()
        else:
            logger.error("Invalid model name")
            return

        # Train Traditional Model
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        logger.info(f"Validation Accuracy: {acc:.4f}")
        logger.info(classification_report(y_val, preds))

    # Save Model
    if save_model_path:
        logger.info(f"Saving model to {save_model_path}")
        torch.save(model, save_model_path)

    # Visuals
    plt.plot(range(epochs), acc)
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
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
                    save_model_path=config.get('VulnScan.train Settings', 'save_model_path'))
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
    for model_main in ["NeuralNetwork", "CNN", "Tfidf_LogReg",
                       "CountVectorizer_LogReg", "RandomForest",
                       "ExtraTrees", "GBM", "XGBoost", "DecisionTree",
                       "NaiveBayes"]:

        code_names = {
            "CountVectorizer_LogReg": "cl",
            "CNN": "cn",
            "DecisionTree": "dt",
            "ExtraTrees": "et",
            "GBM": "g",
            "NeuralNetwork": "n",
            "NaiveBayes": "nb",
            "RandomForest": "r",
            "Tfidf_LogReg": "tl",
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
                        save_model_path=model_path_save)
        except FileNotFoundError as e:
            logger.error(f"File Not Found Error in training model {model_main}: {e}")
            exit(1)
        except AttributeError as e:
            logger.error(f"Attribute Error in training model {model_main}: {e}")
            exit(1)
        except Exception as e:
            logger.error(f"Error in training model {model_main}: {e}")
            exit(1)
