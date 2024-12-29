from __future__ import annotations

import nltk
from Levenshtein import ratio as levenshtein_ratio
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
import difflib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet

import json


class SimilarityScorer:
    weights = {
        "jaccard": 0.1,
        "cosine": 0.25,
        "synonym": 0.25,
        "difflib": 0.1,
        "enhanced": 0.2,
        "levenshtein": 0.1
    }
    history_file = "weights_history.json"

    @staticmethod
    def __jaccard_similarity(str1: str, str2: str) -> float:
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    @staticmethod
    def __cosine_similarity(str1: str, str2: str) -> float:
        vectorizer = CountVectorizer().fit_transform([str1, str2])
        vectors = vectorizer.toarray()
        vec1, vec2 = vectors[0], vectors[1]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def __synonym_similarity(str1: str, str2: str) -> float:
        def get_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())
            return synonyms

        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        score = 0
        for word1 in set1:
            for word2 in set2:
                if word1 == word2 or word2 in get_synonyms(word1):
                    score += 1
        return score / max(len(set1), len(set2))

    @staticmethod
    def __enhanced_similarity(user_input: str, target: str) -> float:
        def preprocess(text: str) -> str:
            lemmatizer = nltk.WordNetLemmatizer()
            return " ".join(lemmatizer.lemmatize(word.lower()) for word in text.split())

        user_input = preprocess(user_input)
        target = preprocess(target)

        vectorizer = TfidfVectorizer().fit_transform([user_input, target])
        tfidf_similarity = (vectorizer * vectorizer.T).A[0, 1]

        set1, set2 = set(user_input.split()), set(target.split())
        jaccard_similarity = len(set1 & set2) / len(set1 | set2)

        def synonyms(word: str) -> set[str]:
            return set(lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas())

        match_score = sum(
            1 for word in user_input.split() if any(word in synonyms(target_word) for target_word in target.split())
        )
        wordnet_score = match_score / len(user_input.split())

        return (tfidf_similarity * 0.4) + (jaccard_similarity * 0.4) + (wordnet_score * 0.2)

    @classmethod
    def calculate_similarity(cls, user_input: str, target: str) -> float:
        jaccard = cls.__jaccard_similarity(user_input, target)
        cosine = cls.__cosine_similarity(user_input, target)
        synonym = cls.__synonym_similarity(user_input, target)
        difflib_ratio = difflib.SequenceMatcher(None, user_input, target).ratio()
        enhanced_similarity_score = cls.__enhanced_similarity(user_input, target)
        levenshtein = levenshtein_ratio(user_input, target)

        return (jaccard * cls.weights["jaccard"]) + (cosine * cls.weights["cosine"]) + \
            (synonym * cls.weights["synonym"]) + (difflib_ratio * cls.weights["difflib"]) + \
            (enhanced_similarity_score * cls.weights["enhanced"]) + (levenshtein * cls.weights["levenshtein"])

    @classmethod
    def adjust_weights(cls, feedback: dict):
        for key, value in feedback.items():
            if key in cls.weights:
                cls.weights[key] += value
                cls.weights[key] = max(0, min(cls.weights[key], 1))  # Ensure weights are between 0 and 1

        cls.save_weights()

    @classmethod
    def save_weights(cls):
        with open(cls.history_file, 'w') as file:
            json.dump(cls.weights, file)

    @classmethod
    def load_weights(cls):
        try:
            with open(cls.history_file, 'r') as file:
                cls.weights = json.load(file)
        except FileNotFoundError:
            cls.save_weights()


# Load weights at the start
SimilarityScorer.load_weights()


def __map_user_desc_to_flag(flags: list[str], descriptions: list[str], user_input: str,
                            threshold: int = 20) -> str | None:
    """
    Maps user input to the best matching flag based on similarity scores.

    Args:
        flags (list[str]): Available flags_list.
        descriptions (list[str]): Descriptions for the flags_list.
        user_input (str): User-provided description or input.
        threshold (int): Minimum similarity percentage to consider a match.

    Returns:
        str | None: The closest matching flag or None if no match exceeds the threshold.
    """
    max_similarity = 0
    best_match = None

    for flag, desc in zip(flags, descriptions):
        similarity = SimilarityScorer.calculate_similarity(user_input, desc)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = flag

    return (
        f"{best_match} (Accurate to {max_similarity * 100:.2f}%)"
        if max_similarity * 100 > threshold
        else None
    )


flags_list = [
    "--default", "--threaded", "--modded", "--depth",
    "--nopy", "--vulnscan-ai", "--minimal", "--performance-check",
    "--debug", "--backup", "--update", "--dev", "--reboot",
    "--shutdown", "--webhook", "--restore"
]

descriptions_list = [
    "Runs Logicytics with default settings and scripts.",
    "Runs Logicytics using threads for parallel execution.",
    "Executes both default and MODS directory scripts.",
    "Executes all default scripts in threading mode for detailed analysis.",
    "Runs non-python scripts for compatibility on devices without Python.",
    "Detects sensitive data in files using AI and logs paths.",
    "Minimal mode for essential scraping with quick scripts.",
    "Measures performance and execution time of scripts.",
    "Runs the debugger to identify issues and generates a bug report log.",
    "Backups Logicytics files to a dedicated directory.",
    "Updates Logicytics from the GitHub repository.",
    "Developer mode for contributors to register their contributions.",
    "Reboots the device after execution.",
    "Shuts down the device after execution.",
    "Sends a zip file via webhook.",
    "Restores Logicytics files from backup."
]

user_inputs = [
    "run with default settings",  # Close to "--default"
    "parallel execution with threads",  # Close to "--threaded"
    "all default scripts in a mods folder",  # Close to "--modded"
    "detailed analysis using threading",  # Close to "--depth"
    "non-python mode for older devices",  # Close to "--nopy"
    "ai scanning for sensitive data",  # Close to "--vulnscan-ai"
    "minimal scraping",  # Close to "--minimal"
    "measure script execution performance",  # Close to "--performance-check"
    "check for issues and bugs",  # Close to "--debug"
    "make a backup of files",  # Close to "--backup"
    "update from github repo",  # Close to "--update"
    "developer contributor mode",  # Close to "--dev"
    "reboot after completing tasks",  # Close to "--reboot"
    "shutdown after running",  # Close to "--shutdown"
    "send zip files via webhook",  # Close to "--webhook"
    "restore files from backup",  # Close to "--restore"
    "something unrelated",  # No close match
]

"""    
for user_input in user_inputs:
    suggestion = __map_user_desc_to_flag(flags_list, descriptions, user_input, threshold=20)
    print(f"User Input: {user_input}")
    print(f"Suggestion: {suggestion}")
    print("-" * 40)
"""

print(SimilarityScorer.weights)
for user_input in user_inputs:
    suggestion = __map_user_desc_to_flag(flags_list, descriptions_list, user_input, threshold=20)
    print(f"User Input: {user_input}")
    print(f"Suggestion: {suggestion}")
    print("-" * 40)
