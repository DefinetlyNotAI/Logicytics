from configparser import ConfigParser

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import os


def load_data(data_paths):
    data = []
    if os.path.isdir(data_paths):
        for root, _, files in os.walk(data_paths):
            for file in files:
                print("Loading File: ", file)
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data.append(f.read())
    else:
        with open(data_paths, 'r', encoding='utf-8') as f:
            data.append(f.read())
    return data


def choose_vectorizer(vectorizer_types):
    print("Vectorizer Type: ", vectorizer_types)
    print("Vectorizing Data...")
    if vectorizer_types == 'tfidf':
        return TfidfVectorizer(max_features=10000)
    elif vectorizer_types == 'count':
        return CountVectorizer(max_features=10000)
    else:
        raise ValueError("Unsupported vectorizer type. Choose 'tfidf' or 'count'.")


def main(data_paths, vectorizer_types, output_paths):
    data = load_data(data_paths)
    vectorizer = choose_vectorizer(vectorizer_types)
    vectorizer.fit(data)
    joblib.dump(vectorizer, os.path.join(output_paths, "Vectorizer.pkl"))
    print(f"Vectorizer saved to {output_paths}")


if __name__ == "__main__":
    print("Reading config file")
    config = ConfigParser()
    config.read('../../config.ini')
    data_path = config.get('VulnScan.vectorizer Settings', 'data_path')
    vectorizer_type = config.get('VulnScan.vectorizer Settings', 'vectorizer_type')
    output_path = config.get('VulnScan.vectorizer Settings', 'output_path')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(data_path, vectorizer_type, output_path)
