import os
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "cves"  # Directory containing .json and .txt files
MODEL_PATH = "vulnerability_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def load_cve(file_path) -> str:
    """
    Load a CVE from a .txt file.
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""


def load_cve_data() -> tuple[list[str], list[int]]:
    """
    Load CVE data from both .json and .txt files in the data directory.
    Labels:
    - 1 for vulnerabilities with "critical" or "high" severity.
    - 0 for others.
    """
    data = []
    labels = []
    for root, dirs, files in os.walk(DATA_DIR):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for sub_root, _, sub_files in os.walk(dir_path):
                for file in sub_files:
                    file_path = os.path.join(sub_root, file)
                    if file.endswith(".txt") or file.endswith(".json"):
                        print(f"Loading {file}...")
                        content = load_cve(file_path)
                        label = 1 if "critical" in file.lower() or "high" in file.lower() else 0
                        data.append(content)
                        labels.append(label)
    return data, labels


def preprocess_text(data) -> Any:
    """
    Convert text data into numerical features using TF-IDF.
    """
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    tfidf_transformer = TfidfTransformer()
    counts = vectorizer.fit_transform(data)
    features = tfidf_transformer.fit_transform(counts)
    joblib.dump(vectorizer, VECTORIZER_PATH)  # Save vectorizer for later use
    return features


def train_model():
    """
    Train a model using the CVE data.
    """
    data, labels = load_cve_data()
    features = preprocess_text(data)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model Training Completed!")

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    input("This may take around 10-20 minutes per 1m files. Press Enter to continue...")
    train_model()
