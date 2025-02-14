from __future__ import annotations

import mimetypes
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import torch
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer

from logicytics import log

# Ignore all warnings
warnings.filterwarnings("ignore")

# Caching dictionaries for model and vectorizer
model_cache = {}
vectorizer_cache = {}

# Locks for thread-safe model and vectorizer loading
model_lock = threading.Lock()
vectorizer_lock = threading.Lock()

# Global variables to hold loaded model and vectorizer
model_to_use = None
vectorizer_to_use = None


def load_model(model_path_to_load: str) -> safe_open | torch.nn.Module:
    """
    Load a machine learning model from the specified file path with caching.

    Parameters:
        model_path_to_load (str): Full file path to the model file to be loaded.

    Returns:
        safe_open | torch.nn.Module: Loaded model object.

    Raises:
        ValueError: If the model file does not have a supported extension (.pkl, .safetensors, or .pth).
    """
    # Check cache first
    if model_path_to_load in model_cache:
        log.info(f"Using cached model from {model_path_to_load}")
        return model_cache[model_path_to_load]

    # Load model if not cached
    if model_path_to_load.endswith('.pkl'):
        model = joblib.load(model_path_to_load)
    elif model_path_to_load.endswith('.safetensors'):
        model = safe_open(model_path_to_load, framework='torch')
    elif model_path_to_load.endswith('.pth'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model = torch.load(model_path_to_load, weights_only=False)
    else:
        raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")

    # Cache the model
    model_cache[model_path_to_load] = model
    log.info(f"Loaded and cached model from {model_path_to_load}")
    return model


def load_vectorizer(vectorizer_path_to_load: str) -> TfidfVectorizer:
    """
    Load the vectorizer from the specified file path with caching.

    Parameters:
        vectorizer_path_to_load (str): Full file path to the vectorizer file to be loaded.

    Returns:
        TfidfVectorizer: Loaded vectorizer object.
    """
    # Check cache first
    if vectorizer_path_to_load in vectorizer_cache:
        log.info(f"Using cached vectorizer from {vectorizer_path_to_load}")
        return vectorizer_cache[vectorizer_path_to_load]

    # Load vectorizer if not cached
    vectorizer = joblib.load(vectorizer_path_to_load)

    # Cache the vectorizer
    vectorizer_cache[vectorizer_path_to_load] = vectorizer
    log.info(f"Loaded and cached vectorizer from {vectorizer_path_to_load}")
    return vectorizer


@log.function
def scan_path(model_path: str, scan_paths: str, vectorizer_path: str):
    """
    Scan a specified path for sensitive content using a pre-trained machine learning model and vectorizer.

    This function handles loading the model and vectorizer if they are not already initialized, and then performs a vulnerability scan on the given path.

    Args:
        model_path (str): Filesystem path to the machine learning model file to be used for scanning.
        scan_paths (str): Filesystem path to the file or directory that will be scanned for sensitive content.
        vectorizer_path (str): Filesystem path to the vectorizer file used for text feature extraction.

    Raises:
        Exception: Captures and logs any errors that occur during the scanning process, preventing the entire scanning operation from halting.

    Side Effects:
        - Loads global model and vectorizer if not already initialized
        - Logs information about model and vectorizer loading
        - Calls vulnscan() to perform actual file scanning
        - Logs any errors encountered during scanning
    """
    global model_to_use, vectorizer_to_use
    try:
        # Load model and vectorizer if not already loaded
        if model_to_use is None:
            with model_lock:
                if model_to_use is None:
                    log.info(f"Loading model from {model_path}")
                    model_to_use = load_model(model_path)

        if vectorizer_to_use is None:
            with vectorizer_lock:
                if vectorizer_to_use is None:
                    log.info(f"Loading vectorizer from {vectorizer_path}")
                    vectorizer_to_use = load_vectorizer(vectorizer_path)

        scan_vulnscan(model_to_use, scan_paths, vectorizer_to_use)
    except FileNotFoundError as err:
        log.error(f"File not found while scanning {scan_paths}: {err}")
    except PermissionError as err:
        log.error(f"Permission denied while scanning {scan_paths}: {err}")
    except (torch.serialization.pickle.UnpicklingError, RuntimeError) as err:
        log.error(f"Model loading failed for {scan_paths}: {err}")
    except Exception as err:
        log.error(f"Unexpected error scanning {scan_paths}: {err}")


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


@log.function
def scan_vulnscan(model, SCAN_PATH, vectorizer):
    """
    Scan a file to determine if it contains sensitive content and log the results.

    Args:
        model (object): Machine learning model used for content sensitivity classification.
        SCAN_PATH (str): Absolute or relative file path to be scanned for sensitive content.
        vectorizer (object): Text vectorization model to transform file content into feature representation.

    Returns:
        None: Logs sensitive file details and appends file path to 'Sensitive_File_Paths.txt' if sensitive content is detected.

    Side Effects:
        - Logs scanning information using the configured logger
        - Creates or appends to 'Sensitive_File_Paths.txt' when sensitive content is found
        - Writes sensitive file paths to the log file

    Raises:
        IOError: If there are issues writing to the 'Sensitive_File_Paths.txt' file
    """
    log.debug(f"Scanning {SCAN_PATH}")
    result, probability, reason = scan_file(model, vectorizer, SCAN_PATH)
    if result:
        log.debug(f"File {SCAN_PATH} is sensitive with probability {probability:.2f}. Reason: {reason}")
        if not os.path.exists("Sensitive_File_Paths.txt"):
            with open("Sensitive_File_Paths.txt", "w") as sensitive_file:
                sensitive_file.write(f"{SCAN_PATH}\n\n")
        with open("Sensitive_File_Paths.txt", "a") as sensitive_file:
            sensitive_file.write(f"{SCAN_PATH}\n")


@log.function
def vulnscan(paths_to_scan: list[str]):
    log.warning(
        "Once again, remember that this script is a proof of concept and should not be used in production. Its in beta and not the most accurate.")
    log.info("Loading Files - This may take some time and consume memory!!")

    try:
        max_workers = min(32, os.cpu_count() * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(os.path.join, root, file_main) for path_to_scan in paths_to_scan for
                       root, _, files_main in
                       os.walk(path_to_scan) for file_main in files_main]
            for future in futures:
                paths.append(future.result())
    except Exception as e:
        log.error(f"Scan failed: {e}")
    finally:
        try:
            model_cache.clear()
            vectorizer_cache.clear()
            model_to_use = None
            vectorizer_to_use = None
            log.info("Scan complete!")
        except Exception as e:
            log.error(f"Scan cleanup failed: {e}")
            log.warning("We recommend restarting your device to free up memory just in case.")

    # Start scanning
    log.warning("Starting scan - This may take hours and consume memory!!")
    log.warning("THIS MAY TAKE HUGE AMOUNTS OF SPACE!!!")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_paths = len(paths)
            completed = 0
            futures = [
                executor.submit(
                    scan_path,
                    "VulnScan/Model SenseMini .3n3.pth",
                    path,
                    "VulnScan/Vectorizer .3n3.pkl"
                )
                for path in paths
            ]
            for future in futures:
                try:
                    future.result()
                    completed += 1
                    if completed % 100 == 0:
                        progress = (completed / total_paths) * 100
                        log.info(f"Scan progress: {progress:.1f}% ({completed}/{total_paths})")
                except Exception as e:
                    log.error(f"Scan failed: {e}")
    except Exception as e:
        log.error(f"Scan failed: {e}")
    finally:
        try:
            model_cache.clear()
            vectorizer_cache.clear()
            model_to_use = None
            vectorizer_to_use = None
            log.info("Scan complete!")
        except Exception as e:
            log.error(f"Scan cleanup failed: {e}")
            log.warning("We recommend restarting your device to free up memory just in case.")


# Now supports importing for custom paths
if __name__ == "__main__":
    # Start scanning
    paths = []
    base_paths = [
        "C:\\Users\\",
        "C:\\Windows\\Logs",
        "C:\\Program Files",
        "C:\\Program Files (x86)"
    ]
    vulnscan(base_paths)
