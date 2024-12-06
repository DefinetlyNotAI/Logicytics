import logging
import os

import joblib
import torch
from safetensors import safe_open as safetensors_load

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_model(model_path):
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_path.endswith('.safetensors'):
        return safetensors_load(model_path, framework='torch')
    elif model_path.endswith('.pth'):
        return torch.load(model_path)
    else:
        raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")


def is_sensitive(model, vectorizer, file_content):
    features = vectorizer.transform([file_content])
    prediction = model.predict(features)
    return prediction[0] == 1


def scan_file(model, vectorizer, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return is_sensitive(model, vectorizer, content)


def scan_directory(model, vectorizer, dir_path):
    results = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                results[file_path] = scan_file(model, vectorizer, file_path)
            except Exception as e:
                logging.error(f"Error scanning file {file_path}: {e}")
    return results


def main(MODEL_PATH, FILE_PATH, VECTORIZER_PATH):
    model = load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)  # Adjust as needed

    if os.path.isfile(FILE_PATH):
        result = scan_file(model, vectorizer, FILE_PATH)
        logging.info(f"File {FILE_PATH} is {'sensitive' if result else 'not sensitive'}")
    elif os.path.isdir(FILE_PATH):
        results = scan_directory(model, vectorizer, FILE_PATH)
        for file_path, is_sensitive_main in results.items():
            logging.info(f"File {file_path} is {'sensitive' if is_sensitive_main else 'not sensitive'}")
    else:
        logging.error("Invalid path provided. Please provide a valid file or directory path.")


if __name__ == "__main__":
    # Test with file - Sensitive
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .1r1\trained_model.pkl", r"C:\Users\Hp\Desktop\Shahm\Password.txt", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .2b1\model.safetensors", r"C:\Users\Hp\Desktop\Shahm\Password.txt", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .2l1\lstm_model.pth", r"C:\Users\Hp\Desktop\Shahm\Password.txt", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")

    # Test with directory - No sensitive files
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .1r1\trained_model.pkl", r"C:\Users\Hp\Desktop\Shahm\Work", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .2b1\model.safetensors", r"C:\Users\Hp\Desktop\Shahm\Work", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")
    main(r"C:\Users\Hp\Desktop\Model Tests\Model Sense\Model Sense .2l1\lstm_model.pth", r"C:\Users\Hp\Desktop\Shahm\Work", r"C:\Users\Hp\Desktop\Model Tests\Model Sense - Vectorizer\Vectorizer.pkl")
