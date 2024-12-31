import gzip
import json
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# File for storing user history data
history_file = 'User_History.json.gz'  # Changed to .gz extension


class FlagMatch:
    @classmethod
    def _generate_summary_and_graph(cls):
        """Generates a full summary and graph based on user history data."""
        # TODO Yet in beta
        # Load the decompressed history data using the load_history function
        if not os.path.exists(history_file):
            exit("No history data found.")

        history_data = cls.load_history()

        # Extract interactions and flag usage count
        interactions = history_data['interactions']
        flags_usage = history_data['flags_usage']

        # Summary of flag usage
        total_interactions = sum(flags_usage.values())

        print("User Interaction Summary:")
        for flag, details in interactions.items():
            print(f"\nFlag: {flag}")

            accuracies = [detail['accuracy'] for detail in details]
            device_names = [detail['device_name'] for detail in details]
            user_inputs = [detail['user_input'] for detail in details]

            average_accuracy = sum(accuracies) / len(accuracies)
            most_common_device = Counter(device_names).most_common(1)[0][0]
            average_user_input = Counter(user_inputs).most_common(1)[0][0]

            print(f"  Average Accuracy: {average_accuracy:.2f}%")
            print(f"  Most Common Device Name: {most_common_device}")
            print(f"  Most Common User Input: {average_user_input}")

        # Print the summary to the console
        print(f"\n\nTotal Interactions with the match flag feature: {total_interactions}")
        print("\nFlag Usage Summary:")
        for flag, count in flags_usage.items():
            print(f"  {flag}: {count} times")

        # Generate the graph for flag usage
        flags = list(flags_usage.keys())
        counts = list(flags_usage.values())

        plt.figure(figsize=(10, 6))
        plt.barh(flags, counts, color='skyblue')
        plt.xlabel('Usage Count')
        plt.title('Flag Usage Frequency')
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # Adjust layout

        # Save and display the graph
        plt.savefig('Flag_usage_summary.png')

    @staticmethod
    def load_history() -> dict[str, any]:
        """Loads the user history from the gzipped JSON file."""
        try:
            with gzip.open(history_file, 'rt', encoding='utf-8') as f:  # Use 'rt' mode for text read
                return json.load(f)
        except FileNotFoundError:
            return {'interactions': [], 'flags_usage': Counter()}

    @staticmethod
    def save_history(history_data: dict[str, any]):
        """Saves the user history to the gzipped JSON file."""
        with gzip.open(history_file, 'wt', encoding='utf-8') as f:  # Use 'wt' mode for text write
            json.dump(history_data, f, indent=4)

    @classmethod
    def update_history(cls, user_input: str, matched_flag: str, accuracy: float):
        """Updates the history based on the user's input and flag match."""
        history_data = cls.load_history()

        # Ensure that interactions is a dictionary (not a list)
        if not isinstance(history_data['interactions'], dict):
            history_data['interactions'] = {}

        # Create a new interaction dictionary
        interaction = {
            'timestamp': datetime.now().strftime('%H:%M:%S - %d/%m/%Y'),
            'user_input': user_input,
            'accuracy': accuracy,
            'device_name': os.getlogin()
        }

        # Ensure the flag exists in the interactions dictionary
        if matched_flag not in history_data['interactions']:
            history_data['interactions'][matched_flag] = []

        # Append the new interaction to the flag's list of interactions
        history_data['interactions'][matched_flag].append(interaction)

        # Ensure the flag exists in the flags_usage counter and increment it
        if matched_flag not in history_data['flags_usage']:
            history_data['flags_usage'][matched_flag] = 0
        history_data['flags_usage'][matched_flag] += 1

        cls.save_history(history_data)

    @classmethod
    def match_flag(cls, user_input: str, flags: list[str], flag_description: list[str]) -> tuple[str, float]:
        """
        Matches user_input to flag_description using advanced semantic similarity.
        Returns the corresponding flag and the accuracy of the match.

        Parameters:
            user_input (str): The input string to match.
            flags (list): List of flags.
            flag_description (list): List of flag descriptions.

        Returns:
            tuple: (matched_flag, accuracy) or ('Nothing matched (Accuracy < 25%)', 0.0).
        """
        if len(flags) != len(flag_description):
            raise ValueError("flags and flag_description lists must be of the same length")

        # Load a pre-trained model for sentence similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and efficient model

        # Combine flags and descriptions for better matching context
        combined_descriptions = [f"{flag} {desc}" for flag, desc in zip(flags, flag_description)]

        # Encode user input and all descriptions
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        descriptions_embeddings = model.encode(combined_descriptions, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(user_embedding, descriptions_embeddings).squeeze(0).tolist()

        # Find the best match
        best_index = max(range(len(similarities)), key=lambda i: similarities[i])
        best_accuracy = similarities[best_index] * 100
        best_match = flags[best_index] if best_accuracy > 25.0 else "Nothing matched (Accuracy < 25%)"

        # Update history
        cls.update_history(user_input, best_match, best_accuracy)

        # TODO If accuracy is low, suggest flags based on historical data
        """    
        if best_accuracy < 25.0:
            suggested_flags = suggest_flags_based_on_history()
            best_match = suggested_flags[0] if suggested_flags else best_match
        """
        return best_match, best_accuracy


# ---------------------------- Test the Updated Function ---------------------------- #

# Flags and descriptions
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

# User inputs for testing
users_inputs = [
    "run with default settings",
    "parallel execution with threads",
    "all scripts in a mods folder",
    "detailed analysis",
    "non-python mode for older devices",
    "ai scanning for sensitive data",
    "minimal scraping",
    "measure script execution performance",
    "check for issues and bugs",
    "make a backup of files",
    "update logicytics",
    "developer mode",
    "restart after completing tasks",
    "shutdown after running",
    "send files via a webhook",
    "restore files from backup",
    "something unrelated",
]

# Test the updated function
for users_input in users_inputs:
    flag_received, accuracy_received = FlagMatch.match_flag(users_input, flags_list, descriptions_list)
    print(f"User input: {users_input}\nMatched flag: {flag_received}\nAccuracy: {accuracy_received:.2f}%\n")
