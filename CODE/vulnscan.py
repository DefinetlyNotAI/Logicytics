from __future__ import annotations

import mimetypes
import os
import threading
import warnings

import joblib
import torch
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Set up logging
from logicytics import Log, DEBUG

# Use v3 models on this! Especially NN models

if __name__ == "__main__":
    log = Log(
        {"log_level": DEBUG}
    )


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
            return torch.load(model_path_to_load)
    else:
        raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")


def is_sensitive(model: torch.nn.Module, vectorizer: TfidfVectorizer, file_content: str) -> tuple[bool, float]:
    """
    Determine if the file content is sensitive using the provided model and vectorizer.

    Args:
        model: Machine learning model.
        vectorizer: Vectorizer to transform file content.
        file_content (str): Content of the file to be analyzed.

    Returns:
        tuple: (True if the content is sensitive, False otherwise, prediction probability).
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
            return prediction.argmax(dim=1).item() == 1, probability
    else:
        features = vectorizer.transform([file_content])
        prediction = model.predict_proba(features)
        probability = prediction.max()
        return model.predict(features)[0] == 1, probability


def scan_file(model: torch.nn.Module, vectorizer: TfidfVectorizer, file_path: str) -> tuple[bool, float]:
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


def scan_directory(model: torch.nn.Module, vectorizer, dir_path: str) -> dict[str, tuple[bool, float]]:
    """
    Scan all files in a directory to determine if they contain sensitive content.

    Args:
        model: Machine learning model.
        vectorizer: Vectorizer to transform file content.
        dir_path (str): Path to the directory to be scanned.

    Returns:
        dict: Dictionary with file paths as keys and (sensitivity, prediction probability) as values.
    """
    results = {}
    for roots, _, files_dir in os.walk(dir_path):
        for file in tqdm(files_dir, desc="Scanning files", unit="file", leave=True):
            file_path = os.path.join(roots, file)
            if file.endswith(('.zip', '.rar', '.7z', '.tar', '.gz', '.tar.gz')):
                continue
            results[file_path] = scan_file(model, vectorizer, file_path)

    return results


def main(MODELS_PATH: str, SCAN_PATH: str, VECTORIZER_PATH: str):
    """
    Main function to load the model and vectorizer, and scan the specified path.
    Saves the paths of sensitive files to a file named "Sensitive_File_Paths.txt".

    Args:
        MODELS_PATH (str): Path to the model file.
        SCAN_PATH (str): Path to the file or directory to be scanned.
        VECTORIZER_PATH (str): Path to the vectorizer file.
    """
    log.info(f"Loading model from {MODELS_PATH}")
    model = load_model(MODELS_PATH)
    log.info(f"Loading vectorizer from {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)  # Adjust as needed
    log.info(f"Scanning {SCAN_PATH}")
    if os.path.isfile(SCAN_PATH):
        result, probability = scan_file(model, vectorizer, SCAN_PATH)
        log.info(f"File {SCAN_PATH} is {'sensitive' if result else 'not sensitive'} with probability {probability:.2f}")
        with open("Sensitive_File_Paths.txt", "w") as sensitive_file:
            sensitive_file.write(f"{SCAN_PATH}\n")
    elif os.path.isdir(SCAN_PATH):
        results = scan_directory(model, vectorizer, SCAN_PATH)
        with open("Sensitive_File_Paths.txt", "w") as sensitive_file:
            for file_path, (is_sensitive_main, probability) in results.items():
                log.info(f"File {file_path} is {'sensitive' if is_sensitive_main else 'not sensitive'} with probability {probability:.2f}")
                if is_sensitive_main:
                    sensitive_file.write(f"{file_path}\n")
    else:
        log.error("Invalid path provided. Please provide a valid file or directory path.")
        exit(1)


def scan_path(model_path: str, scan_paths: str, vectorizer_path: str):
    """
        Scan the specified path using the provided model and vectorizer.

        Args:
            model_path (str): Path to the model file.
            scan_paths (str): Path to the file or directory to be scanned.
            vectorizer_path (str): Path to the vectorizer file.
        """
    main(model_path, scan_paths, vectorizer_path)


log.warning("Starting scan - This may take hours!!")

threads = []
paths = [
    "C:\\Users\\",
    "C:\\Windows\\Logs",
    "C:\\Program Files",
    "C:\\Program Files (x86)"
]

for path in paths:
    thread = threading.Thread(target=scan_path,
                              args=("VulnScan/Model SenseMini .3n3.pth", path, "VulnScan/Vectorizer .3n3.pkl"))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
