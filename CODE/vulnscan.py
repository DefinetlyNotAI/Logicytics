import mimetypes
import os
import warnings

import joblib
import torch
from safetensors import safe_open as safetensors_load
from tqdm import tqdm

# Set up logging
from logicytics import Log, DEBUG

# Use v3 model on this!

if __name__ == "__main__":
    log = Log(
        {"log_level": DEBUG}
    )


def load_model(model_path_to_load):
    if model_path_to_load.endswith('.pkl'):
        return joblib.load(model_path_to_load)
    elif model_path_to_load.endswith('.safetensors'):
        return safetensors_load(model_path_to_load, framework='torch')
    elif model_path_to_load.endswith('.pth'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return torch.load(model_path_to_load)
    else:
        raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")


def is_sensitive(model, vectorizer, file_content):
    if isinstance(model, torch.nn.Module):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        features = vectorizer.transform([file_content])
        model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features.toarray(), dtype=torch.float32).to(device)
            prediction = model(features_tensor)
            return prediction.argmax(dim=1).item() == 1
    else:
        features = vectorizer.transform([file_content])
        prediction = model.predict(features)
        return prediction[0] == 1


def scan_file(model, vectorizer, file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        return is_sensitive(model, vectorizer, content)
    else:
        with open(file_path, 'r', errors='ignore') as file:
            content = file.read()
        return is_sensitive(model, vectorizer, content)


def scan_directory(model, vectorizer, dir_path):
    results = {}
    for roots, _, files_dir in os.walk(dir_path):
        for file in tqdm(files_dir, desc="Scanning files", unit="file"):
            file_path = os.path.join(roots, file)
            if file.endswith(('.zip', '.rar', '.7z', '.tar', '.gz', '.tar.gz')):
                log.warning(f"Skipping compressed file {file_path}")
                continue
            try:
                results[file_path] = scan_file(model, vectorizer, file_path)
            except Exception as e:
                log.error(f"Error scanning file {file_path}: {e}")

    return results


def main(MODELS_PATH, SCAN_PATH, VECTORIZER_PATH):
    log.info(f"Loading model from {MODELS_PATH}")
    model = load_model(MODELS_PATH)
    log.info(f"Loading vectorizer from {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)  # Adjust as needed
    log.info(f"Scanning {SCAN_PATH}")
    if os.path.isfile(SCAN_PATH):
        result = scan_file(model, vectorizer, SCAN_PATH)
        log.info(f"File {SCAN_PATH} is {'sensitive' if result else 'not sensitive'}")
    elif os.path.isdir(SCAN_PATH):
        results = scan_directory(model, vectorizer, SCAN_PATH)
        for file_path, is_sensitive_main in results.items():
            log.info(f"File {file_path} is {'sensitive' if is_sensitive_main else 'not sensitive'}")
    else:
        log.error("Invalid path provided. Please provide a valid file or directory path.")
        exit(1)


if __name__ == "__main__":
    main("VulnScan/Model SenseMini .3n3.pth", "C:\\", "VulnScan/Vectorizer .3n3.pkl")
