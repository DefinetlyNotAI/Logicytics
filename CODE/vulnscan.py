from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
#  add this to config.ini -> max_workers = min(32, os.cpu_count() * 2)
#  add UNREADABLE_EXTENSIONS as well to config.ini

UNREADABLE_EXTENSIONS = [
    ".exe", ".dll", ".so",  # Executables & libraries
    ".zip", ".tar", ".gz", ".7z", ".rar",  # Archives
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",  # Images
    ".mp3", ".wav", ".flac", ".aac", ".ogg",  # Audio
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",  # Video
    ".pdf",  # PDFs aren't plain text
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",  # Microsoft Office files
    ".odt", ".ods", ".odp",  # OpenDocument files
    ".bin", ".dat", ".iso",  # Binary, raw data, disk images
    ".class", ".pyc", ".o", ".obj",  # Compiled code
    ".sqlite", ".db",  # Databases
    ".ttf", ".otf", ".woff", ".woff2",  # Fonts
    ".lnk", ".url"  # Links
]


class _SensitiveDataScanner:
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
        self._load_model()
        self._load_vectorizer()

    def _load_model(self) -> None:
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

    def _load_vectorizer(self) -> None:
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

    def _is_sensitive(self, file_content: str) -> tuple[bool, float, str]:
        """Determines if a file's content is sensitive using the model."""
        if isinstance(self.model, torch.nn.Module):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            # Use sparse matrices to save memory
            features = self.vectorizer.transform([file_content]).tocsr()
            self.model.eval()
            with torch.no_grad():
                # Convert sparse matrix to tensor more efficiently
                features_tensor = torch.sparse_coo_tensor(
                    torch.LongTensor([features.nonzero()[0], features.nonzero()[1]]),
                    torch.FloatTensor(features.data),
                    size=features.shape
                ).to(device)
                prediction = self.model(features_tensor)
                probability = torch.softmax(prediction, dim=1).max().item()
                # Get top features from sparse matrix directly
                feature_scores = features.data
                top_indices = np.argsort(feature_scores)[-5:]
                reason = ", ".join([self.vectorizer.get_feature_names_out()[i] for i in top_indices])
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
            return self._is_sensitive(content)
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
    def __init__(self, model_path: str, vectorizer_path: str):
        self.scanner = _SensitiveDataScanner(model_path, vectorizer_path)

    @log.function
    def scan_directory(self, scan_paths: list[str]) -> None:
        """Scans multiple directories for sensitive files."""
        max_workers = min(32, os.cpu_count() * 2)
        log.debug(f"max_workers={max_workers}")

        log.info("Getting directories files...")
        try:
            # Fast file collection using ThreadPoolExecutor and efficient flattening
            with ThreadPoolExecutor(max_workers=max_workers):
                all_files = []
                for path in scan_paths:
                    try:
                        all_files.extend([str(f) for f in Path(path).rglob('*') if f.is_file()])
                    except Exception as e:
                        log.warning(f"Error collecting files from {path}: {e}")
                        continue  # Skip this path and continue with others

                log.info(f"Files collected successfully: {len(all_files)}")

        except Exception as e:
            log.error(f"Failed to collect files: {e}")
            return

        log.info(f"Scanning {len(all_files)} files...")

        try:
            # Use ThreadPoolExecutor for scanning files concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                total_len_modifiable = len(all_files)

                # Submit scan tasks
                with tqdm(total=total_len_modifiable,
                          desc="\033[32mSCAN\033[0m     \033[94mSubmitting Scan Tasks\033[0m",
                          unit="file", bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}") as submit_pbar:

                    for file in all_files:
                        if any(file.lower().endswith(ext) for ext in UNREADABLE_EXTENSIONS):
                            log.debug(f"Skipping file '{file}'")
                            total_len_modifiable -= 1
                            submit_pbar.update(1)
                            continue

                        futures[executor.submit(self.scanner.scan_file, file)] = file
                        submit_pbar.update(1)

                # Scan progress tracking
                log.info(f"Valid file count: {total_len_modifiable}")
                with tqdm(total=total_len_modifiable, desc="\033[32mSCAN\033[0m     \033[94mScanning Files\033[0m",
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

                        scan_pbar.update(1)

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
    try:
        base_paths = [
            "C:\\Users\\",
            "C:\\Windows\\Logs",
            "C:\\Program Files",
            "C:\\Program Files (x86)"
        ]
        vulnscan = VulnScan("VulnScan/Model SenseMini .3n3.pth", "VulnScan/Vectorizer .3n3.pkl")
        vulnscan.scan_directory(base_paths)
    except KeyboardInterrupt:
        log.warning("User interrupted. Please don't do this as it won't follow the code's cleanup process")
        exit(0)
