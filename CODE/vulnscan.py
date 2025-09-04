import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from logicytics import log, config

# ================== GLOBAL SETTINGS ==================

# File scan settings
TEXT_EXTENSIONS = {
    ".txt", ".log", ".csv", ".json", ".xml", ".html", ".md", ".cfg", ".ini", ".yml", ".yaml",
    ".rtf", ".tex", ".rst", ".adoc", ".properties", ".conf", ".bat", ".ps1", ".sh", ".tsv",
    ".dat", ".env", ".toml", ".dockerfile", ".gitignore", ".gitattributes", ".npmrc", ".editorconfig"
}
MAX_TEXT_LENGTH = config.get("VulnScan Settings", "text_char_limit", fallback=None)
MAX_TEXT_LENGTH = int(MAX_TEXT_LENGTH) if MAX_TEXT_LENGTH not in (None, "None", "") else None
# Threading
NUM_WORKERS = config.get("VulnScan Settings", "max_workers", fallback="auto")
NUM_WORKERS = min(32, (os.cpu_count() or 1) * 2) if NUM_WORKERS == "auto" else int(NUM_WORKERS)
# Classification threshold
SENSITIVE_THRESHOLD = float(
    config.get("VulnScan Settings", "threshold", fallback=0.6))  # Probability cutoff to consider a file sensitive

# Paths
SENSITIVE_PATHS = [
    r"C:\Users\%USERNAME%\Documents",
    r"C:\Users\%USERNAME%\Desktop",
    r"C:\Users\%USERNAME%\Downloads",
    r"C:\Users\%USERNAME%\AppData\Roaming",
    r"C:\Users\%USERNAME%\AppData\Local",
    r"C:\Users\%USERNAME%\OneDrive",
    r"C:\Users\%USERNAME%\Dropbox",
    r"C:\Users\%USERNAME%\Google Drive",
]
SAVE_DIR = r"VulnScan_Files"  # Backup folder
MODEL_PATH = r"vulnscan/Model_SenseMacro.4n1.pth"  # Your trained model checkpoint
REPORT_JSON = "report.json"
REPORT_CSV = "report.csv"

# ================== DEVICE SETUP ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.debug(f"Using device: {DEVICE}")


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
os.makedirs(SAVE_DIR, exist_ok=True)


# ================== FILE PROCESSING ==================
def process_file(filepath):
    try:
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in TEXT_EXTENSIONS:
            return None

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f_:
            content = f_.read()
        if not content.strip():
            return None

        # Limit file length
        if MAX_TEXT_LENGTH is not None:
            content = content[:MAX_TEXT_LENGTH]

        # Split content into lines
        lines = [line_ for line_ in content.splitlines() if line_.strip()]
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
        backup_path = os.path.join(SAVE_DIR, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(filepath, backup_path)

        return {
            "file": filepath,
            "probability": max_prob,
            "copied_to": backup_path,
            "reason": top_lines
        }

    except Exception as e:
        log.error(f"Could not process {filepath}: {e}")
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
def main():
    log.info(f"Scanning directory: {ROOT_DIR} - This will take some time...")
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

    print()
    log.debug("Sensitive files detected and backed up:")
    for entry in sensitive:
        log.debug(f" - {entry['file']} (prob={entry['probability']:.4f})")
        for line in entry["reason"]:
            log.debug(f"     -> {line}")

    print()
    log.info("Backup completed.\n")
    log.debug(f"Files copied into: {SAVE_DIR}")
    log.debug(f"JSON report saved as: {REPORT_JSON}")
    log.debug(f"CSV report saved as: {REPORT_CSV}")


if __name__ == "__main__":
    log.info(f"Starting VulnScan with {NUM_WORKERS} thread workers and {len(SENSITIVE_PATHS)} paths...")
    for path in SENSITIVE_PATHS:
        expanded_path = os.path.expandvars(path)
        if os.path.exists(expanded_path):
            ROOT_DIR = expanded_path
            main()
        else:
            log.warning(f"Path does not exist and will be skipped: {expanded_path}")
