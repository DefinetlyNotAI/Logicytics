import os
import joblib
import pandas as pd
from matplotlib import pyplot as plt

# Paths
MODEL_PATH = "vulnerability_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def extract_features(file_path, vectorizer):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            print(f"Extracting features from {file_path}")
            return vectorizer.transform([content])
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def scan_drive(drive="C:\\"):
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Loading vectorizer from {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)

    results = []
    print(f"Scanning drive {drive} for .txt files")
    for root, dirs, files in os.walk(drive):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".txt"):  # Example: scan only .txt files
                print(f"Scanning file {file_path}")
                features = extract_features(file_path, vectorizer)
                if features is not None:
                    prediction = model.predict(features)[0]
                    print(f"Prediction for {file_path}: {prediction}")
                    results.append({"file_path": file_path, "vulnerability": prediction})

    df_pd = pd.DataFrame(results)
    return df_pd


def visualize_vulnerabilities(df_pd):
    print("Visualizing vulnerabilities")
    vulnerable_files = df_pd[df_pd["vulnerability"] == 1]
    file_sizes = [os.stat(file).st_size for file in vulnerable_files["file_path"]]
    vulnerability_scores = [10] * len(file_sizes)  # Placeholder for vulnerability score

    plt.scatter(file_sizes, vulnerability_scores, c="red", alpha=0.5)
    plt.xlabel("File Size (bytes)")
    plt.ylabel("Vulnerability Score (0-10)")
    plt.title("Vulnerabilities in Text Files")
    plt.show()


if __name__ == "__main__":
    print("Starting drive scan")
    input("This may take around 10-20 minutes per 1m files. Press Enter to continue...")
    df = scan_drive()
    print("Scan complete")
    print(df[df["vulnerability"] == 1])
    visualize_vulnerabilities(df)
    print("Visualization complete")
