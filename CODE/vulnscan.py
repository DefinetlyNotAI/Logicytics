from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from logicytics import log

# Ignore all warnings
warnings.filterwarnings("ignore")


# TODO: v3.4.2
#  apply Batch file reading,
#  Use Asynchronous File Scanning,
#  Optimize Model Loading and Caching,
#  Improve Feature Extraction

# TODO: v3.4.1
#  also add a global variable called MAX_FILE_SIZE, if its none ignore it, else only scan files under that file size (default at 50MB)
#  max_workers = os.cpu_count(),


class SensitiveDataScanner:
    """
    Class for scanning files for sensitive content using a trained model.
    """

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        self.model_cache = {}
        self.vectorizer_cache = {}

        self.model_lock = threading.Lock()
        self.vectorizer_lock = threading.Lock()

        self.model = None
        self.vectorizer = None
        self.load_model()
        self.load_vectorizer()

    def load_model(self) -> None:
        """Loads and caches the ML model."""
        if self.model_path in self.model_cache:
            log.info(f"Using cached model from {self.model_path}")
            self.model = self.model_cache[self.model_path]
            return

        if self.model_path.endswith('.pkl'):
            self.model = joblib.load(self.model_path)
        elif self.model_path.endswith('.safetensors'):
            self.model = safe_open(self.model_path, framework='torch')
        elif self.model_path.endswith('.pth'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = torch.load(self.model_path, weights_only=False)
        else:
            raise ValueError("Unsupported model file format. Use .pkl, .safetensors, or .pth")

        self.model_cache[self.model_path] = self.model
        log.info(f"Loaded and cached model from {self.model_path}")

    def load_vectorizer(self) -> None:
        """Loads and caches the vectorizer."""
        if self.vectorizer_path in self.vectorizer_cache:
            log.info(f"Using cached vectorizer from {self.vectorizer_path}")
            self.vectorizer = self.vectorizer_cache[self.vectorizer_path]
            return

        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
        except Exception as e:
            log.critical(f"Failed to load vectorizer from {self.vectorizer_path}: {e}")
            exit(1)
        self.vectorizer_cache[self.vectorizer_path] = self.vectorizer
        log.info(f"Loaded and cached vectorizer from {self.vectorizer_path}")

    def is_sensitive(self, file_content: str) -> tuple[bool, float, str]:
        """Determines if a file's content is sensitive using the model."""
        if isinstance(self.model, torch.nn.Module):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            features = self.vectorizer.transform([file_content])
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(features.toarray(), dtype=torch.float32).to(device)
                prediction = self.model(features_tensor)
                probability = torch.softmax(prediction, dim=1).max().item()
                top_features = np.argsort(features.toarray()[0])[-5:]
                reason = ", ".join([self.vectorizer.get_feature_names_out()[i] for i in top_features])
                return prediction.argmax(dim=1).item() == 1, probability, reason
        else:
            features = self.vectorizer.transform([file_content])
            prediction = self.model.predict_proba(features)
            probability = prediction.max()
            top_features = np.argsort(features.toarray()[0])[-5:]
            reason = ", ".join([self.vectorizer.get_feature_names_out()[i] for i in top_features])
            return self.model.predict(features)[0] == 1, probability, reason

    def scan_file(self, file_path: str) -> tuple[bool, float, str]:
        """Scans a file for sensitive content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            return self.is_sensitive(content)
        except Exception as e:
            log.error(f"Failed to scan {file_path}: {e}")
            return False, 0.0, "Error reading file"

    def cleanup(self):
        """Clears caches and resets model & vectorizer."""
        self.model_cache.clear()
        self.vectorizer_cache.clear()
        self.model = None
        self.vectorizer = None
        log.info("Cleanup complete!")


class VulnScan:
    """
    Class for scanning directories for sensitive files using SensitiveDataScanner.
    """

    def __init__(self, model_path: str, vectorizer_path: str):
        self.scanner = SensitiveDataScanner(model_path, vectorizer_path)

    @log.function
    def scan_directory(self, scan_paths: list[str]) -> None:
        """Scans multiple directories for sensitive files."""
        max_workers = min(32, os.cpu_count() * 2)

        try:
            # Collect all files to scan using concurrent futures for file discovery
            all_files = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for scan_path in scan_paths:
                    for root, _, files in os.walk(scan_path):
                        for file in files:
                            futures.append(executor.submit(os.path.join, root, file))

                for future in as_completed(futures):
                    all_files.append(future.result())

            log.info(f"Files collected successfully: {len(all_files)}")

        except Exception as e:
            log.error(f"Failed to collect files: {e}")
            return

        log.info(f"Scanning {len(all_files)} files...")

        try:
            # Use ThreadPoolExecutor to scan files concurrently, reducing the overhead
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                # Submit scan tasks with tqdm updating per file
                with tqdm(total=len(all_files), desc="\033[32mINFO\033[0m     \033[94mSubmitting Scan Tasks\033[0m",
                          unit="file", bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}") as submit_pbar:
                    for file in all_files:
                        future = executor.submit(self.scanner.scan_file, file)
                        futures[future] = file
                        submit_pbar.update(1)  # Update after each file is submitted

                # Scan progress tracking
                with tqdm(total=len(all_files), desc="\033[32mINFO\033[0m     \033[94mScanning Files\033[0m",
                          unit="file", bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}") as scan_pbar:
                    sensitive_files = []
                    for future in as_completed(futures):
                        try:
                            result, probability, reason = future.result()
                            if result:
                                file_path = futures[future]
                                log.debug(
                                    f"Sensitive file detected: {file_path} (Confidence: {probability:.2f}). Reason: {reason}")
                            sensitive_files.append(file_path)
                        except Exception as e:
                            log.error(f"Scan failed: {e}")

                        scan_pbar.update(1)  # Update scanning progress

                    # Write all sensitive files at once
                    with open("Sensitive_File_Paths.txt", "a") as sensitive_file:
                        if sensitive_files:
                            sensitive_file.write("\n".join(sensitive_files) + "\n")
                        else:
                            sensitive_file.write("Sadly no sensitive file's were detected.")

        except Exception as e:
            log.error(f"Scanning error: {e}")

        self.scanner.cleanup()


if __name__ == "__main__":
    base_paths = [
        "C:\\Users\\",
        "C:\\Windows\\Logs",
        "C:\\Program Files",
        "C:\\Program Files (x86)"
    ]
    vulnscan = VulnScan("VulnScan/Model SenseMini .3n3.pth", "VulnScan/Vectorizer .3n3.pkl")
    vulnscan.scan_directory(base_paths)
