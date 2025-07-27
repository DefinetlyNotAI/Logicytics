from __future__ import annotations

import asyncio
import os
import threading
import warnings

import aiofiles
import joblib
import numpy as np
# noinspection PyPackageRequirements
import torch
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm

from logicytics import log, config

warnings.filterwarnings("ignore")

UNREADABLE_EXTENSIONS = config.get("VulnScan Settings", "unreadable_extensions").split(",")
MAX_FILE_SIZE_MB = config.get("VulnScan Settings", "max_file_size_mb", fallback="None")
raw_workers = config.get("VulnScan Settings", "max_workers", fallback="auto")
max_workers = min(32, os.cpu_count() * 2) if raw_workers == "auto" else int(raw_workers)

if MAX_FILE_SIZE_MB != "None":
    MAX_FILE_SIZE_MB = max(int(MAX_FILE_SIZE_MB), 1)
else:
    MAX_FILE_SIZE_MB = None


class _SensitiveDataScanner:
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
        with self.model_lock:
            if self.model_path in self.model_cache:
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
                raise ValueError("Unsupported model file format")

            self.model_cache[self.model_path] = self.model

    def _load_vectorizer(self) -> None:
        with self.vectorizer_lock:
            if self.vectorizer_path in self.vectorizer_cache:
                self.vectorizer = self.vectorizer_cache[self.vectorizer_path]
                return

            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
            except Exception as e:
                log.critical(f"Failed to load vectorizer: {e}")
                exit(1)

            self.vectorizer_cache[self.vectorizer_path] = self.vectorizer

    def _extract_features(self, content: str):
        return self.vectorizer.transform([content])

    def _is_sensitive(self, content: str) -> tuple[bool, float, str]:
        features = self._extract_features(content)
        if isinstance(self.model, torch.nn.Module):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            indices = torch.LongTensor(np.vstack(features.nonzero()))
            values = torch.FloatTensor(features.data)
            tensor = torch.sparse_coo_tensor(indices, values, size=features.shape).to(device)

            with torch.no_grad():
                pred = self.model(tensor)
                prob = torch.softmax(pred, dim=1).max().item()
                reason = ", ".join(self.vectorizer.get_feature_names_out()[i] for i in np.argsort(features.data)[-5:])
                return pred.argmax(dim=1).item() == 1, prob, reason
        else:
            probs = self.model.predict_proba(features)
            top_indices = np.argsort(features.toarray()[0])[-5:]
            reason = ", ".join(self.vectorizer.get_feature_names_out()[i] for i in top_indices)
            return self.model.predict(features)[0] == 1, probs.max(), reason

    async def scan_file_async(self, file_path: str) -> tuple[bool, float, str]:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            return self._is_sensitive(content)
        except Exception as e:
            log.error(f"Failed to scan {file_path}: {e}")
            return False, 0.0, "Error"

    def cleanup(self):
        self.model_cache.clear()
        self.vectorizer_cache.clear()
        self.model = None
        self.vectorizer = None
        log.info("Cleanup complete.")


class VulnScan:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.scanner = _SensitiveDataScanner(model_path, vectorizer_path)

    @log.function
    def scan_directory(self, scan_paths: list[str]) -> None:
        log.info("Collecting files...")
        all_files = []

        for path in scan_paths:
            try:
                all_files.extend(str(f) for f in Path(path).rglob('*') if f.is_file())
                log.debug(f"Found {len(all_files)} files in {path}")
            except Exception as e:
                log.warning(f"Skipping path {path} due to error: {e}")

        log.info(f"Collected {len(all_files)} files.")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._async_scan(all_files))

    async def _async_scan(self, files: list[str]) -> None:
        valid_files = []

        for file in files:
            try:
                file_size_mb = os.path.getsize(file) / (1024 * 1024)
                if MAX_FILE_SIZE_MB and file_size_mb > MAX_FILE_SIZE_MB:
                    continue
                if any(file.lower().endswith(ext) for ext in UNREADABLE_EXTENSIONS):
                    continue
                valid_files.append(file)
            except Exception as e:
                log.debug(f"Skipping file {file}: {e}")

        log.info(f"Valid files to scan: {len(valid_files)}")

        semaphore = asyncio.Semaphore(max_workers)
        sensitive_files = []

        async def scan_worker(scan_file):
            async with semaphore:
                result, prob, reason = await self.scanner.scan_file_async(scan_file)
                if result:
                    log.debug(f"SENSITIVE: {scan_file} | Confidence: {prob:.2f} | Reason: {reason}")
                    sensitive_files.append(scan_file)

        tasks = [scan_worker(f) for f in valid_files]

        with tqdm(total=len(valid_files), desc="\033[32mSCAN\033[0m     \033[94mScanning Files\033[0m",
                  unit="file", bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}\n") as pbar:
            for f in asyncio.as_completed(tasks):
                await f
                pbar.update(1)

        with open("Sensitive_File_Paths.txt", "a") as out:
            out.write("\n".join(sensitive_files) + "\n" if sensitive_files else "No sensitive files detected.\n")

        self.scanner.cleanup()


if __name__ == "__main__":
    try:
        base_paths = [
            "C:\\Users\\",
            "C:\\Windows\\Logs",
            "C:\\Program Files",
            "C:\\Program Files (x86)"
        ]
        vulnscan = VulnScan("vulnscan/SenseMini.3n3.pth", "vulnscan/vectorizer.3n3.pkl")
        vulnscan.scan_directory(base_paths)
    except KeyboardInterrupt:
        log.warning("User interrupted. Exiting gracefully.")
        exit(0)
