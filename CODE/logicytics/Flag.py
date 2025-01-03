from __future__ import annotations

import argparse
import configparser
import difflib
import gzip
import json
import os
from collections import Counter
from datetime import datetime

# Check if the script is being run directly, if not, set up the library
if __name__ == '__main__':
    exit("This is a library, Please import rather than directly run.")
else:
    # Set up constants and configurations
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
    except FileNotFoundError:
        try:
            config.read('../config.ini')
        except FileNotFoundError:
            exit("No configuration file found.")
    # Save user preferences?
    SAVE_PREFERENCES = config.getboolean("Settings", "save_preferences")
    # Debug mode for Sentence Transformer
    DEBUG_MODE = config.getboolean("Flag Settings", "model_debug")  # Debug mode for Sentence Transformer
    # File for storing user history data
    HISTORY_FILE = 'logicytics/User_History.json.gz'  # User history file
    # Minimum accuracy threshold for flag suggestions
    MIN_ACCURACY_THRESHOLD = float(
        config.get("Flag Settings", "accuracy_min"))  # Minimum accuracy threshold for flag suggestions
    if not 0 <= MIN_ACCURACY_THRESHOLD <= 100:
        raise ValueError("accuracy_min must be between 0 and 100")


class Match:
    @staticmethod
    def __get_sim(user_input: str, all_descriptions: list[str]) -> list[float]:
        """
        Compute cosine similarity between user input and flag descriptions using a Sentence Transformer model.
        
        This method encodes the user input and historical flag descriptions into embeddings and calculates their cosine similarities. It handles model loading, logging configuration, and error handling for the embedding process.
        
        Parameters:
            user_input (str): The current user input to match against historical descriptions
            all_descriptions (list[str]): A list of historical flag descriptions to compare
        
        Returns:
            list[float]: A list of similarity scores between the user input and each historical description
        
        Raises:
            SystemExit: If there is an error loading the specified Sentence Transformer model
        
        Notes:
            - Uses the model specified in the configuration file
            - Configures logging based on the global DEBUG_MODE setting
            - Converts embeddings to tensors for efficient similarity computation
        """
        # Encode the current user input and historical inputs
        from sentence_transformers import SentenceTransformer, util

        import logging  # Suppress logging messages from Sentence Transformer due to verbosity
        # Set the logging level based on the debug mode, either DEBUG or ERROR (aka only important messages)
        if DEBUG_MODE:
            logging.getLogger("sentence_transformers").setLevel(logging.DEBUG)
        else:
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        try:
            MODEL = SentenceTransformer(config.get("Flag Settings", "model_to_use"))
        except Exception as e:
            print(f"Error: {e}")
            print("Please check the model name in the config file.")
            print(f"Model name {config.get('Flag Settings', 'model_to_use')} may not be valid.")
            exit(1)

        user_embedding = MODEL.encode(user_input, convert_to_tensor=True, show_progress_bar=DEBUG_MODE)
        historical_embeddings = MODEL.encode(all_descriptions, convert_to_tensor=True, show_progress_bar=DEBUG_MODE)

        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(user_embedding, historical_embeddings).squeeze(0).tolist()
        return similarities

    @classmethod
    def __suggest_flags_based_on_history(cls, user_input: str) -> list[str]:
        """
        Suggests flags based on historical data and similarity to the current input.
        
        This method analyzes historical user interactions to recommend relevant flags when preferences for saving history are enabled. It uses semantic similarity to find the most contextually related flags from past interactions.
        
        Parameters:
            user_input (str): The current input for which suggestions are needed.
        
        Returns:
            list[str]: A list of suggested flags derived from historical interactions, filtered by similarity threshold.
        
        Notes:
            - Returns an empty list if history saving is disabled or no interaction history exists
            - Uses cosine similarity with a minimum threshold of 0.3 to filter suggestions
            - Limits suggestions to top 3 most similar historical inputs
            - Removes duplicate flag suggestions
        """
        if not SAVE_PREFERENCES:
            return []
        history_data = cls.load_history()
        if not history_data or 'interactions' not in history_data:
            return []

        interactions = history_data['interactions']
        all_descriptions = []
        all_flags = []

        # Combine all flags and their respective user inputs
        for flag, details in interactions.items():
            all_flags.extend([flag] * len(details))
            all_descriptions.extend([detail['user_input'] for detail in details])

        # Encode the current user input and historical inputs
        # Compute cosine similarities
        similarities = cls.__get_sim(user_input, all_descriptions)

        # Find the top 3 most similar historical inputs
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
        suggested_flags = [all_flags[i] for i in top_indices if similarities[i] > 0.3]

        # Remove duplicates and return suggestions
        return list(dict.fromkeys(suggested_flags))

    @classmethod
    def _generate_summary_and_graph(cls):
        """
        Generates a comprehensive summary and visualization of user interaction history with command-line flags.
        
        This method processes historical interaction data, computes statistical insights, and creates a bar graph representing flag usage frequency. It performs the following key tasks:
        - Loads historical interaction data from a compressed file
        - Calculates and prints detailed statistics for each flag
        - Generates a horizontal bar graph of flag usage counts
        - Saves the graph visualization to a PNG file
        
        Parameters:
            cls (Match): The class instance containing historical data methods
        
        Raises:
            SystemExit: If no history data file is found
            FileNotFoundError: If unable to save the graph in default locations
        
        Side Effects:
            - Prints detailed interaction summary to console
            - Saves flag usage graph as a PNG image
            - Uses matplotlib to create visualization
        
        Notes:
            - Currently in beta stage of development
            - Requires matplotlib for graph generation
            - Attempts to save graph in multiple predefined directory paths
        """
        # TODO Yet in beta
        # Load the decompressed history data using the load_history function
        import matplotlib.pyplot as plt

        if not os.path.exists(HISTORY_FILE):
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
        try:
            plt.savefig('../ACCESS/DATA/Flag_usage_summary.png')
            print("\nFlag Usage Summary Graph saved to 'ACCESS/DATA/Flag_usage_summary.png'")
        except FileNotFoundError:
            try:
                plt.savefig('../../ACCESS/DATA/Flag_usage_summary.png')
                print("\nFlag Usage Summary Graph saved to 'ACCESS/DATA/Flag_usage_summary.png'")
            except FileNotFoundError:
                plt.savefig('Flag_usage_summary.png')
                print("\nFlag Usage Summary Graph saved in current working directory as 'Flag_usage_summary.png'")

    @staticmethod
    def load_history() -> dict[str, any]:
        """
        Load user interaction history from a gzipped JSON file.
        
        This method attempts to read and parse historical interaction data from a compressed JSON file. If the file is not found, it returns an empty history structure with an empty interactions dictionary and a zero-initialized flags usage counter.
        
        Returns:
            dict[str, any]: A dictionary containing:
                - 'interactions': A dictionary of past user interactions
                - 'flags_usage': A Counter object tracking flag usage frequencies
        
        Raises:
            json.JSONDecodeError: If the JSON file is malformed
            gzip.BadGzipFile: If the gzipped file is corrupted
        """
        try:
            with gzip.open(HISTORY_FILE, 'rt', encoding='utf-8') as f:  # Use 'rt' mode for text read
                return json.load(f)
        except FileNotFoundError:
            return {'interactions': {}, 'flags_usage': Counter()}

    @staticmethod
    def save_history(history_data: dict[str, any]):
        """
        Save user interaction history to a gzipped JSON file.
        
        This method writes the user history to a compressed JSON file only if saving preferences are enabled. 
        The history is saved with an indentation of 4 spaces for readability.
        
        Parameters:
            history_data (dict[str, any]): A dictionary containing user interaction history data to be saved.
        
        Notes:
            - Saves only if SAVE_PREFERENCES is True
            - Uses gzip compression to reduce file size
            - Writes in UTF-8 encoding
            - Indents JSON for human-readable format
        """
        if SAVE_PREFERENCES:
            with gzip.open(HISTORY_FILE, 'wt', encoding='utf-8') as f:  # Use 'wt' mode for text write
                json.dump(history_data, f, indent=4)

    @classmethod
    def update_history(cls, user_input: str, matched_flag: str, accuracy: float):
        """
        Update the user interaction history with details of a matched flag.
        
        This method records user interactions with flags, including timestamp, input, match accuracy,
        and device information. It only updates history if save preferences are enabled.
        
        Parameters:
            user_input (str): The original input text provided by the user.
            matched_flag (str): The flag that was successfully matched to the user input.
            accuracy (float): The similarity/match accuracy score for the flag.
        
        Side Effects:
            - Modifies the history JSON file by adding a new interaction entry
            - Increments the usage count for the matched flag
            - Requires write access to the history file
        
        Notes:
            - Skips history update if SAVE_PREFERENCES is False
            - Creates new flag entries in history if they do not exist
            - Uses current timestamp and logged-in user's device name
        """
        if not SAVE_PREFERENCES:
            return
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
    def flag(cls, user_input: str, flags: list[str], flag_description: list[str]) -> tuple[str, float]:
        """
        Matches user input to flag descriptions using advanced semantic similarity.
        
        Computes the best matching flag based on cosine similarity between the user input and flag descriptions. 
        Handles matching with a minimum accuracy threshold and provides flag suggestions from historical data 
        if no direct match is found.
        
        Parameters:
            user_input (str): The input string to match against available flags.
            flags (list[str]): List of available command flags.
            flag_description (list[str]): Corresponding descriptions for each flag.
        
        Returns:
            tuple[str, float]: A tuple containing:
                - The best matched flag (or 'Nothing matched')
                - Accuracy percentage of the match (0.0-100.0)
        
        Raises:
            ValueError: If the number of flags and descriptions do not match.
        
        Side Effects:
            - Updates user interaction history
            - Prints flag suggestions if no direct match is found
            - Requires a global MIN_ACCURACY_THRESHOLD to be defined
        
        Example:
            matched_flag, accuracy = Flag.flag("show help", 
                                               ["-h", "--verbose"], 
                                               ["Display help", "Enable verbose output"])
        """
        if len(flags) != len(flag_description):
            raise ValueError("flags and flag_description lists must be of the same length")

        # Combine flags and descriptions for better matching context
        combined_descriptions = [f"{flag} {desc}" for flag, desc in zip(flags, flag_description)]

        # Encode user input and all descriptions
        # Compute cosine similarities
        similarities = cls.__get_sim(user_input, combined_descriptions)

        # Find the best match
        best_index = max(range(len(similarities)), key=lambda i: similarities[i])
        best_accuracy = similarities[best_index] * 100
        best_match = flags[best_index] if best_accuracy > MIN_ACCURACY_THRESHOLD else "Nothing matched"

        # Update history
        cls.update_history(user_input, best_match, best_accuracy)

        # Suggest flags if accuracy is low
        if best_accuracy < MIN_ACCURACY_THRESHOLD:
            suggested_flags = cls.__suggest_flags_based_on_history(user_input)
            if suggested_flags:
                print(f"No Flags matched so suggestions based on historical data: "
                      f"{', '.join(suggested_flags)}")

        return best_match, best_accuracy


class Flag:
    @classmethod
    def __colorify(cls, text: str, color: str) -> str:
        The existing docstring is comprehensive and follows good documentation practices. It clearly explains the function's purpose, parameters, return value, and provides context about the color codes. Therefore:
        
        KEEP_EXISTING
        colors = {
            "y": "\033[93m",
            "r": "\033[91m",
            "b": "\033[94m"
        }
        RESET = "\033[0m"
        return f"{colors.get(color, '')}{text}{RESET}" if color in colors else text

    @classmethod
    def __available_arguments(cls) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
        """
        Defines and parses command-line arguments for the Logicytics application.
        
        This method creates an ArgumentParser with a comprehensive set of flags for customizing the application's behavior. It supports various execution modes, debugging options, system management flags, and post-execution actions.
        
        The method handles argument parsing, provides helpful descriptions for each flag, and includes color-coded hints for user guidance. It also supports suggesting valid flags if an unknown flag is provided.
        
        Parameters:
            cls (type): The class context in which the method is called.
        
        Returns:
            tuple[argparse.Namespace, argparse.ArgumentParser]: A tuple containing:
                - Parsed command-line arguments (Namespace)
                - The configured argument parser object
        """
        # Define the argument parser
        parser = argparse.ArgumentParser(
            description="Logicytics, The most powerful tool for system data analysis. "
                        "This tool provides a comprehensive suite of features for analyzing system data, "
                        "including various modes for different levels of detail and customization.",
            allow_abbrev=False
        )

        # Define Actions Flags
        parser.add_argument(
            "--default",
            action="store_true",
            help="Runs Logicytics with its default settings and scripts. "
                 f"{cls.__colorify('- Recommended for most users -', 'b')}",
        )

        parser.add_argument(
            "--threaded",
            action="store_true",
            help="Runs Logicytics using threads, where it runs in parallel, default settings though"
                 f"{cls.__colorify('- Recommended for some users -', 'b')}",
        )

        parser.add_argument(
            "--modded",
            action="store_true",
            help="Runs the normal Logicytics, as well as any File in the MODS directory, "
                 "Used for custom scripts as well as default ones.",
        )

        parser.add_argument(
            "--depth",
            action="store_true",
            help="This flag will run all default script's in threading mode, "
                 "as well as any clunky and huge code, which produces a lot of data "
                 f"{cls.__colorify('- Will take a long time -', 'y')}",
        )

        parser.add_argument(
            "--nopy",
            action="store_true",
            help="Run Logicytics using all non-python scripts, "
                 f"These may be {cls.__colorify('outdated', 'y')} "
                 "and not the best, use only if the device doesnt have python installed.",
        )

        parser.add_argument(
            "--vulnscan-ai",
            action="store_true",
            help="Run's Logicytics new Sensitive data Detection AI, its a new feature that will "
                 "detect any files that are out of the ordinary, and logs their path. Runs threaded."
                 f"{cls.__colorify('- Beta Mode -', 'y')} "
                 f"{cls.__colorify('- Will take a long time -', 'y')}",
        )

        parser.add_argument(
            "--minimal",
            action="store_true",
            help="Run Logicytics in minimal mode. Just bare essential scraping using only quick scripts",
        )

        parser.add_argument(
            "--performance-check",
            action="store_true",
            help="Run's Logicytics default while testing its performance and time, "
                 "this then shows a table with the file names and time to executed. "
                 f"{cls.__colorify('- Beta Mode -', 'y')}"
        )

        # Define Side Flags
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Runs the Debugger, Will check for any issues, "
                 "warning etc, useful for debugging and issue reporting "
                 f"{cls.__colorify('- Use to get a special log file to report the bug -', 'b')}.",
        )

        parser.add_argument(
            "--backup",
            action="store_true",
            help="Backup Logicytics files to the ACCESS/BACKUPS directory "
                 f"{cls.__colorify('- Use on your own device only -', 'y')}.",
        )

        parser.add_argument(
            "--update",
            action="store_true",
            help="Update Logicytics from GitHub, only if you have git properly installed "
                 "and the project was downloaded via git "
                 f"{cls.__colorify('- Use on your own device only -', 'y')}.",
        )

        parser.add_argument(
            "--dev",
            action="store_true",
            help="Run Logicytics developer mod, this is only for people who want to "
                 "register their contributions properly. "
                 f"{cls.__colorify('- Use on your own device only -', 'y')}.",
        )

        # Define After-Execution Flags
        parser.add_argument(
            "--reboot",
            action="store_true",
            help="Execute Flag that will reboot the device afterward",
        )

        parser.add_argument(
            "--shutdown",
            action="store_true",
            help="Execute Flag that will shutdown the device afterward",
        )

        # Not yet Implemented
        parser.add_argument(
            "--webhook",
            action="store_true",
            help="Execute Flag that will send zip File via webhook "
                 f"{cls.__colorify('- Not yet Implemented -', 'r')}",
        )

        parser.add_argument(
            "--restore",
            action="store_true",
            help="Restore Logicytics files from the ACCESS/BACKUPS directory "
                 f"{cls.__colorify('- Use on your own device only -', 'y')} "
                 f"{cls.__colorify('- Not yet Implemented -', 'r')}",
        )

        # Parse the arguments
        args, unknown = parser.parse_known_args()
        valid_flags = [action.dest for action in parser._actions if action.dest != 'help']
        if unknown:
            cls.__suggest_flag(unknown[0], valid_flags)
            exit(1)
        return args, parser

    @staticmethod
    def __exclusivity_logic(args: argparse.Namespace) -> bool:
        """
        Validates the mutual exclusivity of command-line flags to prevent invalid flag combinations.
        
        This method checks for conflicting or mutually exclusive flags across three flag categories:
        - Special flags (reboot, shutdown, webhook)
        - Action flags (default, threaded, modded, minimal, nopy, depth, performance_check)
        - Exclusive flags (vulnscan_ai)
        
        Parameters:
            args (argparse.Namespace): Parsed command-line arguments to validate.
        
        Returns:
            bool: True if any special flags are set, False otherwise.
        
        Raises:
            SystemExit: If incompatible flag combinations are detected, with an error message describing the conflict.
        """
        special_flags = {
            args.reboot,
            args.shutdown,
            args.webhook
        }
        action_flags = {
            args.default,
            args.threaded,
            args.modded,
            args.minimal,
            args.nopy,
            args.depth,
            args.performance_check
        }
        exclusive_flags = {
            args.vulnscan_ai,
        }

        if any(special_flags) and not any(action_flags):
            print("Invalid combination of flags_list: Special and Action flag exclusivity issue.")
            exit(1)

        if any(exclusive_flags) and any(action_flags):
            print("Invalid combination of flags_list: Exclusive and Action flag exclusivity issue.")
            exit(1)

        if any(exclusive_flags) and any(special_flags):
            print("Invalid combination of flags_list: Exclusive and Special flag exclusivity issue.")
            exit(1)

        return any(special_flags)

    @staticmethod
    def __used_flags_logic(args: argparse.Namespace) -> tuple[str, ...]:
        """
        Determines the flags that are set to True in the provided command-line arguments.
        
        This method examines the arguments namespace and returns a tuple of flag names 
        that have been activated. It limits the returned flags to a maximum of two to 
        prevent excessive flag usage.
        
        Parameters:
            args (argparse.Namespace): Parsed command-line arguments to be analyzed.
        
        Returns:
            tuple[str, ...]: A tuple containing the names of flags set to True, 
            with a maximum of two flags.
        
        Notes:
            - If no flags are set, returns an empty tuple.
            - Stops collecting flags after finding two True flags to limit complexity.
        """
        flags = {key: getattr(args, key) for key in vars(args)}
        true_keys = []
        for key, value in flags.items():
            if value:
                true_keys.append(key)
                if len(true_keys) == 2:
                    break
        return tuple(true_keys)

    @classmethod
    def __suggest_flag(cls, user_input: str, valid_flags: list[str]):
        """
        Suggests the closest valid flag based on the user's input and provides interactive flag matching.
        
        This method handles flag suggestion through two mechanisms:
        1. Using difflib to find close flag matches
        2. Prompting user for a description to find the most relevant flag
        
        Args:
            user_input (str): The flag input by the user.
            valid_flags (list[str]): The list of valid flags.
        
        Behavior:
            - If a close flag match exists, suggests the closest match
            - If no close match, prompts user for a description
            - Uses the Match.flag method to find the most accurate flag based on description
            - Prints matching results, with optional detailed output in debug mode
        
        Side Effects:
            - Prints suggestions and matched flags to console
            - Prompts user for additional input if no direct match is found
        """
        # Get the closest valid flag match based on the user's input
        closest_matches = difflib.get_close_matches(user_input, valid_flags, n=1, cutoff=0.6)
        if closest_matches:
            print(f"Invalid flag '{user_input}', Did you mean '--{closest_matches[0]}'?")

        # Prompt the user for a description if no close match is found
        user_input_desc = input("We can't find a match, Please provide a description: ").lower()

        # Map the user-provided description to the closest valid flag
        flags_list = [f"--{flag}" for flag in valid_flags]
        descriptions_list = [f"Run Logicytics with {flag}" for flag in valid_flags]
        flag_received, accuracy_received = Match.flag(user_input_desc, flags_list, descriptions_list)
        if DEBUG_MODE:
            print(f"User input: {user_input_desc}\nMatched flag: {flag_received}\nAccuracy: {accuracy_received:.2f}%\n")
        else:
            print(f"Matched flag: {flag_received} (Accuracy: {accuracy_received:.2f}%)\n")

    @staticmethod
    def show_help_menu(return_output: bool = False):
        """
        Display the help menu for the Logicytics application.
        
        This method retrieves the argument parser from the Flag class and either prints or returns the help text based on the input parameter.
        
        Args:
            return_output (bool, optional): Controls the method's behavior. 
                - If True, returns the formatted help text as a string. 
                - If False (default), prints the help text directly to the console.
        
        Returns:
            str or None: Help text as a string if return_output is True, otherwise None.
        
        Example:
            # Print help menu to console
            Flag.show_help_menu()
        
            # Get help menu as a string
            help_text = Flag.show_help_menu(return_output=True)
            print(help_text)
        """
        parser = Flag.__available_arguments()[1]
        if return_output:
            return parser.format_help()
        else:
            parser.print_help()

    @classmethod
    def data(cls) -> tuple[str, str | None]:
        """
        Handles the parsing and validation of command-line flags.
        
        This method processes command-line arguments, validates their usage, and manages flag interactions. It ensures that:
        - Only one primary action flag is used at a time
        - Special flags are handled with specific logic
        - No invalid flag combinations are permitted
        - User history is optionally updated based on preferences
        
        Parameters:
            cls (type): The class context for accessing class methods
        
        Returns:
            tuple[str, str | None]: A tuple containing:
                - The primary matched flag
                - An optional secondary flag (None if not applicable)
                - Exits the program if no flags are used or invalid combinations are detected
        
        Raises:
            SystemExit: Terminates the program with an error message for:
                - Invalid flag combinations
                - No flags specified
        """
        args, parser = cls.__available_arguments()
        special_flag_used = cls.__exclusivity_logic(args)

        used_flags = [flag for flag in vars(args) if getattr(args, flag)]

        if not special_flag_used and len(used_flags) > 1:
            print("Invalid combination of flags: Maximum 1 action flag allowed.")
            exit(1)

        if special_flag_used:
            used_flags = cls.__used_flags_logic(args)
            if len(used_flags) > 2:
                print("Invalid combination of flags: Maximum 2 flag mixes allowed.")
                exit(1)

        if not used_flags:
            cls.show_help_menu()
            exit(0)

        # Update history with the matched flag(s)
        if not SAVE_PREFERENCES:
            return

        def update_data_history(matched_flag: str):
            """
            Update the usage count for a specific flag in the user's interaction history.
            
            This method increments the usage count for a given flag in the historical data. If the flag
            does not exist in the history, it initializes its count to 0 before incrementing.
            
            Parameters:
                matched_flag (str): The flag whose usage count needs to be updated.
            
            Side Effects:
                - Modifies the 'flags_usage' dictionary in the user's history file
                - Saves the updated history data to a persistent storage
            
            Example:
                update_data_history('--verbose')  # Increments usage count for '--verbose' flag
            """
            history_data = Match.load_history()
            # Ensure the flag exists in the flags_usage counter and increment it
            if matched_flag not in history_data['flags_usage']:
                history_data['flags_usage'][matched_flag] = 0
            history_data['flags_usage'][matched_flag] += 1
            Match.save_history(history_data)

        if len(used_flags) == 2:
            for flag in used_flags:
                update_data_history(flag)
            return tuple(used_flags)
        update_data_history(used_flags[0])
        return used_flags[0], None
