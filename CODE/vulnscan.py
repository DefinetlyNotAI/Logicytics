from __future__ import annotations

import mimetypes
import os
import threading
import warnings

import joblib
import numpy as np
import torch
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

log.info("Locking threads - Model and Vectorizer")
model_lock = threading.Lock()
vectorizer_lock = threading.Lock()

model_to_use = None
vectorizer_to_use = None


def load_model(model_path_to_load: str) -> safe_open | torch.nn.Module:
    """
    Load a machine learning model from the specified file path.

    Args:
        model_path_to_load (str): Path to the model file.

    Returns:
        model: Loaded model object.

    Raises:
        ValueError: If the model file format is unsupported.
    """
    if model_path_to_load.endswith('.pkl'):
        return joblib.load(model_path_to_load)
    elif model_path_to_load.endswith('.safetensors'):
        return safe_open(model_path_to_load, framework='torch')
    elif model_path_to_load.endswith('.pth'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return torch.load(model_path_to_load, weights_only=False)
    else:
        raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")


def scan_path(model_path: str, scan_paths: str, vectorizer_path: str):
    global model_to_use, vectorizer_to_use
    try:
        with model_lock:
            if model_to_use is None:
                log.info(f"Loading model from {model_path}")
                model_to_use = load_model(model_path)
        with vectorizer_lock:
            if vectorizer_to_use is None:
                log.info(f"Loading vectorizer from {vectorizer_path}")
                vectorizer_to_use = joblib.load(vectorizer_path)
        vulnscan(model_to_use, scan_paths, vectorizer_to_use)
    except Exception as e:
        log.error(f"Error scanning path {scan_paths}: {e}")


def is_sensitive(model: torch.nn.Module, vectorizer: TfidfVectorizer, file_content: str) -> tuple[bool, float, str]:
    """
    Determine if the file content is sensitive using the provided model and vectorizer.

    Args:
        model: Machine learning model.
        vectorizer: Vectorizer to transform file content.
        file_content (str): Content of the file to be analyzed.

    Returns:
        tuple: (True if the content is sensitive, False otherwise, prediction probability, reason).
    """
    if isinstance(model, torch.nn.Module):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        features = vectorizer.transform([file_content])
        model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features.toarray(), dtype=torch.float32).to(device)
            prediction = model(features_tensor)
            probability = torch.softmax(prediction, dim=1).max().item()
            top_features = np.argsort(features.toarray()[0])[-5:]
            reason = ", ".join([vectorizer.get_feature_names_out()[i] for i in top_features])
            return prediction.argmax(dim=1).item() == 1, probability, reason
    else:
        features = vectorizer.transform([file_content])
        prediction = model.predict_proba(features)
        probability = prediction.max()
        top_features = np.argsort(features.toarray()[0])[-5:]
        reason = ", ".join([vectorizer.get_feature_names_out()[i] for i in top_features])
        return model.predict(features)[0] == 1, probability, reason


def scan_file(model: torch.nn.Module, vectorizer: TfidfVectorizer, file_path: str) -> tuple[bool, float, str]:
    """
    Scan a single file to determine if it contains sensitive content.

    Args:
        model: Machine learning model.
        vectorizer: Vectorizer to transform file content.
        file_path (str): Path to the file to be scanned.

    Returns:
        tuple: (True if the file is sensitive, False otherwise, prediction probability).
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        return is_sensitive(model, vectorizer, content)
    else:
        with open(file_path, 'r', errors='ignore') as file:
            content = file.read()
        return is_sensitive(model, vectorizer, content)


def vulnscan(model, SCAN_PATH, vectorizer):
    log.info(f"Scanning {SCAN_PATH}")
    result, probability, reason = scan_file(model, vectorizer, SCAN_PATH)
    if result:
        log.info(f"File {SCAN_PATH} is sensitive with probability {probability:.2f}. Reason: {reason}")
        if not os.path.exists("Sensitive_File_Paths.txt"):
            with open("Sensitive_File_Paths.txt", "w") as sensitive_file:
                sensitive_file.write(f"{SCAN_PATH}\n\n")
        with open("Sensitive_File_Paths.txt", "a") as sensitive_file:
            sensitive_file.write(f"{SCAN_PATH}\n")


# Start scanning
log.info("Getting paths to scan - This may take some time!!")

threads = []
paths = []
base_paths = [
    "C:\\Users\\",
    "C:\\Windows\\Logs",
    "C:\\Program Files",
    "C:\\Program Files (x86)"
]

for base_path in base_paths:
    for root, dirs, files_main in os.walk(base_path):
        for file_main in files_main:
            paths.append(os.path.join(root, file_main))

# Start scanning
log.warning("Starting scan - This may take hours and consume memory!!")

for path in paths:
    thread = threading.Thread(target=scan_path,
                              args=("VulnScan/Model SenseMini .3n3.pth", path, "VulnScan/Vectorizer .3n3.pkl"))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
