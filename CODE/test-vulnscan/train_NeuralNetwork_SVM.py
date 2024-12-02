import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import os

EPOCHS: int = 5  # Set number of epochs
FOLDER_PATH: str = r"C:\Users\Hp\Desktop\Model Tests\Model Data\Artificial Generated Data 1M files with 10KB"
MODEL: str = "svm"  # Change `model_type` to "nn" for a neural network model or "svm" for a support vector machine model


# Function to load and preprocess files
def load_data(file_paths):
    data = []
    labels = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Feature extraction (excluding file name)
                word_count = len(content.split())
                special_char_count = sum(c in "@#$%^&*" for c in content)
                sensitive_keywords = content.lower().count("password") + content.lower().count("confidential")
                features = [word_count, special_char_count, sensitive_keywords]
                data.append(features)
                # Label: 1 for sensitive, 0 for non-sensitive
                labels.append(1 if "sensitive" in content.lower() else 0)
                print(f"File {file_path} loaded successfully. Features: {features}, Label: {labels[-1]}")
        except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return np.array(data), np.array(labels)


# Function to save training progress graph
def save_progress_graph(accuracies, filename=f"training_progress_{MODEL}.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


# Function to train the model with hyperparameter tuning
def train_model(file_paths, model_type="svm"):
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
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
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
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters: {grid_search.best_params_}")

    # Train with the best model
    accuracies = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

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
        save_progress_graph(accuracies)

        # Save checkpoint model after every epoch
        if epoch % 1 == 0:
            joblib.dump(best_model, f"trained_model_epoch_{epoch + 1}_{MODEL}.pkl")
            print(f"Model checkpoint saved: trained_model_epoch_{epoch + 1}_{MODEL}.pkl")

    # Save final model
    joblib.dump(best_model, f"trained_model_{MODEL}.pkl")
    print(f"Final model saved as trained_model_{MODEL}.pkl")
    print("Training complete.")


# Main function
if __name__ == "__main__":
    FILE_PATH: list[str] = []
    for file in os.listdir(FOLDER_PATH):
        if file.endswith(".txt"):
            print(f"Indexed file: {file}")
            FILE_PATH.append(os.path.join(FOLDER_PATH, file))
    print(f"Total Indexed file's: {len(FILE_PATH)}")

    if not FILE_PATH:
        print(f"No files found for training. Please ensure '{FOLDER_PATH}' contains text files.")
    else:
        train_model(FILE_PATH, model_type=MODEL)
