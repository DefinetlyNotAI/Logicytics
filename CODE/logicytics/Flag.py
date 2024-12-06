from __future__ import annotations

import argparse
from argparse import ArgumentParser


class Flag:
    @classmethod
    def colorify(cls, text: str, color: str) -> str:
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
                        "including various modes for different levels of detail and customization."
        )

        # Define Actions Flags
        parser.add_argument(
            "--default",
            action="store_true",
            help="Runs Logicytics with its default settings and scripts. "
                 f"{cls.colorify('- Recommended for most users -', 'b')}",
        )

        parser.add_argument(
            "--threaded",
            action="store_true",
            help="Runs Logicytics using threads, where it runs in parallel, default settings though"
                 f"{cls.colorify('- Recommended for some users -', 'b')}",
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
                 f"{cls.colorify('- Will take a long time -', 'y')}",
        )

        parser.add_argument(
            "--nopy",
            action="store_true",
            help="Run Logicytics using all non-python scripts, "
                 f"These may be {cls.colorify('outdated', 'y')} "
                 "and not the best, use only if the device doesnt have python installed.",
        )

        parser.add_argument(
            "--vulnscan-ai",
            action="store_true",
            help="Run's Logicytics new Sensitive data Detection AI, its a new feature that will "
                 "detect any files that are out of the ordinary, and logs their path. Runs threaded."
                 f"{cls.colorify('- Beta Mode -', 'y')} "
                 f"{cls.colorify('- Will take a long time -', 'y')}",
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
                 f"{cls.colorify('- Beta Mode -', 'y')}"
        )

        # Define Side Flags
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Runs the Debugger, Will check for any issues, "
                 "warning etc, useful for debugging and issue reporting "
                 f"{cls.colorify('- Use to get a special log file to report the bug -', 'b')}.",
        )

        parser.add_argument(
            "--backup",
            action="store_true",
            help="Backup Logicytics files to the ACCESS/BACKUPS directory "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )

        parser.add_argument(
            "--update",
            action="store_true",
            help="Update Logicytics from GitHub, only if you have git properly installed "
                 "and the project was downloaded via git "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )

        parser.add_argument(
            "--unzip-extra",
            action="store_true",
            help="Unzip the extra directory zip File "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
        )

        parser.add_argument(
            "--extra",
            action="store_true",
            help="Open's the extra directory menu to use more tools. "
                 f"{cls.colorify('- Still experimental -', 'y')} "
                 f"{cls.colorify('- MUST have used --unzip-extra flag -', 'b')}.",
        )

        parser.add_argument(
            "--dev",
            action="store_true",
            help="Run Logicytics developer mod, this is only for people who want to "
                 "register their contributions properly. "
                 f"{cls.colorify('- Use on your own device only -', 'y')}.",
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
                 f"{cls.colorify('- Not yet Implemented -', 'r')}",
        )

        parser.add_argument(
            "--restore",
            action="store_true",
            help="Restore Logicytics files from the ACCESS/BACKUPS directory "
                 f"{cls.colorify('- Use on your own device only -', 'y')} "
                 f"{cls.colorify('- Not yet Implemented -', 'r')}",
        )

        return parser.parse_args(), parser

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
    def data(cls) -> ArgumentParser | tuple[str]:
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

        if not tuple(used_flags):
            return parser
        else:
            return tuple(used_flags)
