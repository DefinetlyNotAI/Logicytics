from __future__ import annotations

import os
from configparser import ConfigParser
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader

# NN seems to be the best choice for this task

# Set up logging
from logicytics import Log, DEBUG

logger = Log(
    {"log_level": DEBUG,
     "filename": "../../../ACCESS/LOGS/VulnScan_Train.log",
     "colorlog_fmt_parameters":
         "%(log_color)s%(levelname)-8s%(reset)s %(yellow)s%(asctime)s %(blue)s%(message)s",
     }
)


# Dataset Class for PyTorch models
class SensitiveDataDataset(Dataset):
    """
    A custom Dataset class for handling sensitive data for PyTorch models.

    Attributes:
        texts (list[str]): List of text data.
        labels (list[int]): List of labels corresponding to the text data.
        tokenizer (callable, optional): A function to tokenize the text data.
    """

    def __init__(self,
                 texts_init: list[str],
                 labels_init: list[int],
                 tokenizer: Optional[callable] = None):
        """
        Initializes the SensitiveDataDataset with texts, labels, and an optional tokenizer.

        Args:
            texts_init (list[str]): List of text data.
            labels_init (list[int]): List of labels corresponding to the text data.
            tokenizer (callable, optional): A function to tokenize the text data.
        """
        self.texts = texts_init
        self.labels = labels_init
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a sample and its label from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the tokenized text tensor and the label tensor.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            text = self.tokenizer(text)
        return torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def vectorize_text_data(X_trains: list[str], X_vals: list[str], save_model_path: str):
    """
    Vectorizes the text data using TfidfVectorizer and saves the vectorizer model.

    Args:
        X_trains (list[str]): List of training text data.
        X_vals (list[str]): List of validation text data.
        save_model_path (str): Path to save the vectorizer model.

    Returns:
        tuple: Transformed training and validation data as arrays.
    """
    vectorizers = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    joblib.dump(vectorizers, os.path.join(os.path.dirname(save_model_path), 'Vectorizer.pkl'))
    return vectorizers.fit_transform(X_trains).toarray(), vectorizers.transform(X_vals).toarray()


def save_and_plot_model(model: nn.Module,
                        save_model_path: str,
                        accuracy_list: list[float],
                        loss_list: list[float],
                        epochs: int,
                        model_name: str):
    """
    Saves the trained model and plots the accuracy and loss over epochs.

    Args:
        model (nn.Module): The trained PyTorch model.
        save_model_path (str): The path to save the model.
        accuracy_list (list[float]): List of accuracy values over epochs.
        loss_list (list[float]): List of loss values over epochs.
        epochs (int): The number of epochs.
        model_name (str): The name of the model.
    """
    logger.info(f"Saving {model_name} model")
    if save_model_path:
        logger.info(f"Saving model to {save_model_path}.pth")
        torch.save(model, save_model_path + ".pth")

    logger.info(f"Plotting {model_name} model - Accuracy Over Epochs")
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(1, epochs + 1)), accuracy_list, label="Accuracy")
    plt.title(f'{model_name} - Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(save_model_path), f"Model Accuracy Over Epochs - {model_name}.png"))
    plt.show()

    logger.info(f"Plotting {model_name} model - Loss Over Epochs")
    plt.plot(list(range(1, epochs + 1)), loss_list, label="Loss")
    plt.title(f'{model_name} - Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(save_model_path), f"Model Loss Over Epochs - {model_name}.png"))
    plt.show()


def select_model_from_traditional(model_name: str,
                                  epochs: int) -> LogisticRegression | RandomForestClassifier | ExtraTreesClassifier | GradientBoostingClassifier | DecisionTreeClassifier | MultinomialNB | Any:
    """
    Selects and returns a machine learning model based on the provided model name.

    Args:
        model_name (str): The name of the model to select.
        epochs (int): The number of epochs for training (used for LogisticRegression).

    Returns:
        A machine learning model instance corresponding to the model name.
    """
    logger.info(f"Selecting {model_name} model")
    if model_name == 'LogisticRegression':
        return LogisticRegression(max_iter=epochs)
    if model_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100)
    if model_name == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=100)
    if model_name == 'GBM':
        return GradientBoostingClassifier(n_estimators=100)
    if model_name == 'XGBoost':
        return xgb.XGBClassifier(eval_metric='logloss')
    if model_name == 'DecisionTree':
        return DecisionTreeClassifier()
    if model_name == 'NaiveBayes':
        return MultinomialNB()
    if model_name == 'LogReg':
        return LogisticRegression(max_iter=epochs)
    logger.error(f"Invalid model name: {model_name}")
    exit(1)


def train_traditional_model(model_name: str,
                            epochs: int,
                            save_model_path: str):
    """
    Trains a traditional machine learning model.

    Args:
        model_name (str): The name of the model to train.
        epochs (int): The number of epochs for training.
        save_model_path (str): The path to save the trained model.
    """
    global vectorizer, X_val, X_train
    logger.info(f"Using Vectorizer TfidfVectorizer for {model_name} model")
    # Ensure X_train and X_val are lists of strings
    X_train = [str(text) for text in X_train]
    X_val = [str(text) for text in X_val]

    # Call the vectorize_text_data function
    X_train, X_val = vectorize_text_data(X_train, X_val, save_model_path)

    logger.info(f"Training {model_name} model")
    model = select_model_from_traditional(model_name, epochs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy_list = accuracy_score(y_val, predictions)
    logger.info(f"Validation Accuracy: {accuracy_list:.4f}")
    logger.info(classification_report(y_val, predictions))

    loss_list, acc_plot = [], []

    logger.info(f"Training {model_name} model for {epochs} epochs")
    for epoch in range(epochs):
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy_list = accuracy_score(y_val, predictions)
        acc_plot.append(accuracy_list)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy_list:.4f}")
        logger.info(classification_report(y_val, predictions, zero_division=0))

        if hasattr(model, 'predict_proba'):
            loss = model.score(X_val, y_val)
            logger.debug(f"Epoch {epoch + 1}: Model loss: {loss}")
        else:
            loss = 1 - accuracy_list
            logger.debug(f"Epoch {epoch + 1}: Model loss: {loss}")
        loss_list.append(loss)

    save_and_plot_model(model, save_model_path, acc_plot, loss_list, epochs, model_name)


def train_neural_network(epochs: int,
                         batch_size: int,
                         learning_rate: float,
                         save_model_path: str,
                         use_cuda: Optional[bool] = False):
    """
    Trains a neural network model.

    Args:
        epochs (int): The number of epochs to train the model.
        batch_size (int): The size of the batches for training.
        learning_rate (float): The learning rate for the optimizer.
        save_model_path (str): The path to save the trained model.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to False.
    """
    if use_cuda is None:
        use_cuda = False
    global vectorizer, X_val, X_train, labels
    logger.info("Vectorizing text data for Neural Network")
    # Ensure X_train and X_val are lists of strings
    X_train = [str(text) for text in X_train]
    X_val = [str(text) for text in X_val]

    # Call the vectorize_text_data function
    X_train, X_val = vectorize_text_data(X_train, X_val, save_model_path)

    logger.info("Training Neural Network model")
    model = nn.Sequential(nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Linear(128, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.01)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Training on hardware: {device}")
    model.to(device)

    logger.info("Creating DataLoaders for Neural Network")
    train_dataset = SensitiveDataDataset(X_train, y_train)
    val_dataset = SensitiveDataDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    accuracy_list = []
    loss_list = []

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
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            logger.debug(f"Epoch {epoch + 1}: Correct: {correct}, Total: {total}")

        scheduler.step()

        accuracy_list.append(correct / total)
        loss_list.append(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr}")
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {(correct / total):.4f}")

    logger.info("Validating Neural Network model")
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            logger.debug(f"Validation: Correct: {val_correct}, Total: {val_total}")

    val_acc = val_correct / val_total
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    save_and_plot_model(model, save_model_path, accuracy_list, loss_list, epochs, 'NeuralNetwork')


def train_model(
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        save_model_path: str,
        use_cuda: Optional[bool] = False,
):
    """
    Trains a machine learning model based on the specified parameters.

    Args:
        model_name (str): The name of the model to train.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The size of the batches for training.
        learning_rate (float): The learning rate for the optimizer.
        save_model_path (str): The path to save the trained model.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to False.
    """
    if use_cuda is None:
        use_cuda = False
    if model_name == 'NeuralNetwork':
        train_neural_network(epochs, batch_size, learning_rate, save_model_path, use_cuda)
    else:
        train_traditional_model(model_name, epochs, save_model_path)


if __name__ == "__main__":
    # Config file reading and setting constants
    logger.info("Reading config file")
    config = ConfigParser()
    config.read('../../config.ini')
    MODEL_NAME = config.get('VulnScan.train Settings', 'model_name')
    TRAINING_PATH = config.get('VulnScan.train Settings', 'train_data_path')
    EPOCHS = int(config.get('VulnScan.train Settings', 'epochs'))
    BATCH_SIZE = int(config.get('VulnScan.train Settings', 'batch_size'))
    LEARN_RATE = float(config.get('VulnScan.train Settings', 'learning_rate'))
    CUDA = config.getboolean('VulnScan.train Settings', 'use_cuda')
    SAVE_PATH = config.get('VulnScan.train Settings', 'save_model_path')

    # Load Data
    logger.info(f"Loading data from {TRAINING_PATH}")
    texts, labels = [], []
    for filename in os.listdir(TRAINING_PATH):
        with open(os.path.join(config.get('VulnScan.train Settings', 'train_data_path'), filename), 'r',
                  encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(1 if '-sensitive' in filename else 0)
        logger.debug(f"Loaded data from {filename} with label {labels[-1]}")

    # Split Data
    logger.info("Splitting data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(texts,
                                                      labels,
                                                      test_size=0.2,
                                                      random_state=42)

    # Train Model
    try:
        train_model(model_name=MODEL_NAME,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARN_RATE,
                    save_model_path=SAVE_PATH,
                    use_cuda=CUDA)
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
    raise ImportError("This file cannot be imported")
