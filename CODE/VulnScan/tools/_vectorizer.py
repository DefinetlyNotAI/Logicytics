from __future__ import annotations

from configparser import ConfigParser

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import os


def load_data(data_paths: str | os.PathLike) -> list[str]:
    """
    Load data from the specified path(s).

    Args:
        data_paths (str | os.PathLike): Path to a directory or a file containing data.

    Returns:
        list[str]: List of strings, each representing the content of a file.
    """
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


def choose_vectorizer(vectorizer_types: str) -> TfidfVectorizer | CountVectorizer:
    """
    Choose and return a vectorizer based on the specified type.

    Args:
        vectorizer_types (str): Type of vectorizer to use ('tfidf' or 'count').

    Returns:
        TfidfVectorizer | CountVectorizer: The chosen vectorizer.

    Raises:
        ValueError: If an unsupported vectorizer type is specified.
    """
    print("Vectorizer Type: ", vectorizer_types)
    print("Vectorizing Data...")
    if vectorizer_types == 'tfidf':
        return TfidfVectorizer(max_features=10000)
    if vectorizer_types == 'count':
        return CountVectorizer(max_features=10000)
    raise ValueError("Unsupported vectorizer type. Choose 'tfidf' or 'count'.")


def main(data_paths: str, vectorizer_types: str, output_paths: str):
    """
    Main function to load data, choose a vectorizer, fit the vectorizer to the data, and save the vectorizer.

    Args:
        data_paths (str): Path to the data.
        vectorizer_types (str): Type of vectorizer to use ('tfidf' or 'count').
        output_paths (str): Path to save the fitted vectorizer.
    """
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
