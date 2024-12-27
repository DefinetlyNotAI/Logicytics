from __future__ import annotations

import argparse
import difflib
import json
import os


class Flag:
    CONFIG_FILE = 'flag_suggestions_config.json'

    @classmethod
    def __colorify(cls, text: str, color: str) -> str:
        """
        Adds color to the given text based on the specified color code.

        Args:
            text (str): The text to be colorized.
            color (str): The color code ('y' for yellow, 'r' for red, 'b' for blue).

        Returns:
            str: The colorized text if the color code is valid, otherwise the original text.
        """
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
        A static method used to parse command-line arguments for the Logicytics application.

        It defines various flags that can be used to customize the behavior of the application,
        including options for running in default or minimal mode, unzipping extra files,
        backing up or restoring data, updating from GitHub, and more.

        The method returns a tuple containing the parsed arguments and the argument parser object.

        Returns:
            tuple[argparse.Namespace, argparse.ArgumentParser]: A tuple containing the parsed arguments and the argument parser object.
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

        args = parser.parse_args()
        return args, parser

    @staticmethod
    def __exclusivity_logic(args: argparse.Namespace) -> bool:
        """
        Checks if exclusive flags are used in the provided arguments.

        Args:
            args (argparse.Namespace): The arguments to be checked.

        Returns:
            bool: True if exclusive flags are used, False otherwise.
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
            print("Invalid combination of flags: Special and Action flag exclusivity issue.")
            exit(1)

        if any(exclusive_flags) and any(action_flags):
            print("Invalid combination of flags: Exclusive and Action flag exclusivity issue.")
            exit(1)

        if any(exclusive_flags) and any(special_flags):
            print("Invalid combination of flags: Exclusive and Special flag exclusivity issue.")
            exit(1)

        return any(special_flags)

    @staticmethod
    def __used_flags_logic(args: argparse.Namespace) -> tuple[str, ...]:
        """
        Sets flags based on the provided arguments.

        Args:
            args (argparse.Namespace): The arguments to be checked for flags.

        Returns:
            tuple[str, ...]: A tuple of flag names that are set to True.
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
    def data(cls) -> tuple[str, str | None]:
        """
        Handles the parsing and validation of command-line flags.

        Returns either a tuple of used flag names or an ArgumentParser instance.
        """
        args, parser = cls.__available_arguments()
        special_flag_used = cls.__exclusivity_logic(args)

        if not special_flag_used:
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 1:
                print("Invalid combination of flags: Maximum 1 action flag allowed.")
                exit(1)

        if special_flag_used:
            used_flags = cls.__used_flags_logic(args)
            if len(used_flags) > 2:
                print("Invalid combination of flags: Maximum 2 flag mixes allowed.")
                exit(1)

        if len(used_flags) == 2:
            return tuple(used_flags)
        return used_flags[0], None

    @staticmethod
    def show_help_menu(format_output: bool = False):
        """
        Displays the help menu for the Logicytics application.

        Args:
             format_output (bool): If True, returns the help text instead of printing it
        """
        _, parser = Flag.__available_arguments()
        if format_output:
            return parser.format_help()
        else:
            parser.print_help()

    @classmethod
    def load_config(cls):
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r') as file:
                return json.load(file)
        return {}

    @classmethod
    def save_config(cls, config):
        with open(cls.CONFIG_FILE, 'w') as file:
            json.dump(config, file, indent=4)

    @classmethod
    def get_closest_flag(cls, input_flag, available_flags, threshold=0.6):
        closest_matches = difflib.get_close_matches(input_flag, available_flags, n=1, cutoff=threshold)
        if closest_matches:
            return closest_matches[0], difflib.SequenceMatcher(None, input_flag, closest_matches[0]).ratio()
        return None, 0

    @classmethod
    def suggest_flag_based_on_description(cls, description, available_flags):
        """
        Suggests a flag based on the provided description using a simple keyword matching mechanism.

        Args:
            description (str): The description of what the user wants to do.
            available_flags (list[str]): The list of available flags.

        Returns:
            str: The suggested flag based on the description.
        """
        description = description.lower()
        for flag in available_flags:
            if flag in description:
                return flag
        return available_flags[0]

    @classmethod
    def handle_invalid_flag(cls, input_flag, available_flags):
        # Handle the invalid flag and collect feedback
        closest_flag, accuracy = cls.get_closest_flag(input_flag, available_flags)
        if closest_flag:
            print(f"Did you mean '{closest_flag}'? (Accuracy: {accuracy:.2f})")
            feedback = input(f"Was this suggestion helpful? (yes/no): ").strip().lower()
            feedback = 'positive' if feedback == 'yes' else 'negative'
            cls.track_user_interaction(input_flag, closest_flag, feedback)
        else:
            description = input("Flag not recognized. Please describe what you want to do: ")
            suggested_flag = cls.suggest_flag_based_on_description(description, available_flags)
            print(f"Based on your description, you might want to use the '{suggested_flag}' flag.")
            feedback = input(f"Was this suggestion helpful? (yes/no): ").strip().lower()
            feedback = 'positive' if feedback == 'yes' else 'negative'
            cls.track_user_interaction(input_flag, suggested_flag, feedback)

    @classmethod
    def track_user_interaction(cls, input_flag, suggested_flag, feedback):
        # Tracks user interaction and feedback on flag suggestions
        config = cls.load_config()
        if input_flag not in config:
            config[input_flag] = []
        config[input_flag].append({'suggested_flag': suggested_flag, 'feedback': feedback})
        cls.save_config(config)

    @classmethod
    def improve_suggestions(cls):
        # Improve the suggestion logic based on the feedback received
        config = cls.load_config()
        improved_config = {}

        for input_flag, suggestions in config.items():
            # Aggregate feedback for each suggested flag
            feedback_counter = {}
            for suggestion in suggestions:
                suggested_flag = suggestion['suggested_flag']
                feedback = suggestion['feedback']
                if suggested_flag not in feedback_counter:
                    feedback_counter[suggested_flag] = {'positive': 0, 'negative': 0}
                if feedback == 'positive':
                    feedback_counter[suggested_flag]['positive'] += 1
                else:
                    feedback_counter[suggested_flag]['negative'] += 1

            # Determine the best suggestion based on feedback
            best_suggestion = max(feedback_counter,
                                  key=lambda k: feedback_counter[k]['positive'] - feedback_counter[k]['negative'])
            improved_config[input_flag] = best_suggestion

        # Save improved suggestions to the configuration file
        cls.save_config(improved_config)
        print("Improved suggestions config:", improved_config)
