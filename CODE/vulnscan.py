"""from __future__ import annotations

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

UNREADABLE_EXTENSIONS = config.get("VulnScan Settings", "unreadable_extensions").split(
    ","
)
MAX_FILE_SIZE_MB = config.get("VulnScan Settings", "max_file_size_mb", fallback="None")
raw_workers = config.get("VulnScan Settings", "max_workers", fallback="auto")
max_workers = min(32, os.cpu_count() * 2) if raw_workers == "auto" else int(raw_workers)

if MAX_FILE_SIZE_MB != "None":
    MAX_FILE_SIZE_MB = max(int(MAX_FILE_SIZE_MB), 1)
else:
    MAX_FILE_SIZE_MB = None
"""
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

# ================== GLOBAL SETTINGS ==================
# Paths
ROOT_DIR = r"C:\Users\Hp\Desktop\Shahm"  # Folder to scan
BACKUP_DIR = r"C:\Users\Hp\Desktop\VulnScan_Files"  # Backup folder
MODEL_PATH = r"vulnscan/Model_SenseMacro.4n1.pth"  # Your trained model checkpoint

# File scan settings
TEXT_EXTENSIONS = {".txt", ".log", ".csv", ".json", ".xml", ".html", ".md", ".cfg", ".ini", ".yml", ".yaml"}
MAX_TEXT_LENGTH = 1000000  # Max characters per file to scan

# Threading
NUM_WORKERS = 8  # Number of parallel threads for scanning

# Classification threshold
SENSITIVE_THRESHOLD = 0.5  # Probability cutoff to consider a file sensitive

# Reports
REPORT_JSON = os.path.join(os.getcwd(), "report.json")
REPORT_CSV = os.path.join(os.getcwd(), "report.csv")

# ================== DEVICE SETUP ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ================== MODEL DEFINITION ==================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.fc(x)


# ================== LOAD MODELS ==================
# Load classifier
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = SimpleNN(input_dim=384)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# Make backup folder
os.makedirs(BACKUP_DIR, exist_ok=True)


# ================== FILE PROCESSING ==================
def process_file(filepath):
    try:
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in TEXT_EXTENSIONS:
            return None

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if not content.strip():
            return None

        # Limit file length
        content = content[:MAX_TEXT_LENGTH]

        # Split content into lines
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return None

        # Embed all lines
        embeddings = embed_model.encode(lines, convert_to_tensor=True, device=DEVICE)

        # Predict per line
        probs = []
        for emb in embeddings:
            with torch.no_grad():
                output = model(emb.unsqueeze(0))
                probs.append(torch.sigmoid(output).item())

        max_prob = max(probs)
        if max_prob < SENSITIVE_THRESHOLD:
            return None

        # Get top 5 lines contributing most
        top_lines = [lines[i] for i, p in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]]

        # Backup file
        rel_path = os.path.relpath(filepath, ROOT_DIR)
        backup_path = os.path.join(BACKUP_DIR, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(filepath, backup_path)

        return {
            "file": filepath,
            "probability": max_prob,
            "copied_to": backup_path,
            "reason": top_lines
        }

    except Exception as e:
        print(f"[ERROR] Could not process {filepath}: {e}")
    return None


# ================== DIRECTORY SCAN ==================
def scan_directory(root):
    sensitive_files = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for dirpath, _, filenames in os.walk(root):
            for file in filenames:
                futures.append(executor.submit(process_file, os.path.join(dirpath, file)))

        for future in as_completed(futures):
            result = future.result()
            if result:
                sensitive_files.append(result)

    return sensitive_files


# ================== MAIN ==================
if __name__ == "__main__":
    print(f"Scanning directory: {ROOT_DIR}")
    sensitive = scan_directory(ROOT_DIR)

    # Save JSON report
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(sensitive, f, indent=2, ensure_ascii=False)

    # Save CSV report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "probability", "copied_to", "reason"])
        writer.writeheader()
        for entry in sensitive:
            # Join top lines as single string for CSV
            entry_csv = entry.copy()
            entry_csv["reason"] = " | ".join(entry["reason"])
            writer.writerow(entry_csv)

    print("\nSensitive files detected and backed up:")
    for entry in sensitive:
        print(f" - {entry['file']} (prob={entry['probability']:.4f})")
        for line in entry["reason"]:
            print(f"     -> {line}")

    print(f"\nBackup completed.\nFiles copied into: {BACKUP_DIR}")
    print(f"Reports saved as:\n - {REPORT_JSON}\n - {REPORT_CSV}")
