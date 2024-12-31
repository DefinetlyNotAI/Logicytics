from sentence_transformers import SentenceTransformer, util


def match_flag(user_input: str, flags: list[str], flag_description: list[str]) -> tuple[str, float]:
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

    return best_match, best_accuracy


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
    flag_received, accuracy_received = match_flag(users_input, flags_list, descriptions_list)
    print(f"User input: {users_input}\nMatched flag: {flag_received}\nAccuracy: {accuracy_received:.2f}%\n")
