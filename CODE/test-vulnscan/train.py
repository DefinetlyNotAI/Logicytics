import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os


# Function to load and preprocess files
def load_data(file_paths):
    data = []
    labels = []

    for file_path_ld in file_paths:
        try:
            with open(file_path_ld, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Example feature extraction
                word_count = len(content.split())
                special_char_count = sum(c in "@#$%^&*" for c in content)
                sensitive_keywords = content.lower().count("password") + content.lower().count("confidential")
                features = [word_count, special_char_count, sensitive_keywords]
                data.append(features)
                # Label: 1 for sensitive, 0 for non-sensitive
                labels.append(1 if "sensitive" in content.lower() else 0)
                print(f"File {file_path_ld} loaded successfully. Features: {features}, Label: {labels[-1]}")
        except Exception as e:
            print(f"Error reading file {file_path_ld}: {e}")

    return np.array(data), np.array(labels)


# Function to train the model
def train_model(file_paths):
    print("Training model...")

    # Load data
    data, labels = load_data(file_paths)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    epochs = 20  # Set number of epochs
    accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train model
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
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label="Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Progress")
        plt.legend()
        plt.grid()
        plt.savefig("training_progress.png")
        plt.close()

        # Save model checkpoint
        joblib.dump(model, f"trained_model_epoch_{epoch + 1}.pkl")
        print(f"Model checkpoint saved: trained_model_epoch_{epoch + 1}.pkl")

    # Save final model
    joblib.dump(model, "trained_model.pkl")
    print("Final model saved as trained_model.pkl")
    print("Training complete.")


# Main function
if __name__ == "__main__":
    folder_path = r"C:\Users\Hp\Desktop\Model Tests\Model Sense.1L\generated_data_1m-files_10KB"
    file_path = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            os.path.join(folder_path, file)
            print(f"Indexed file: {file}")
            file_path.append(os.path.join(folder_path, file))
    print(f"Total Indexed file's: {len(file_path)}")

    if not file_path:
        print(f"No files found for training. Please ensure '{folder_path}' contains text files.")
    else:
        print(f"Found {len(file_path)} files for training.")
        train_model(file_path)
