import subprocess
import argparse
import json
from Libs.__lib_log import Log


class Actions:
    @staticmethod
    def run_command(command):
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process.stdout
    # TODO Remove this and make others use command normally - TEST THEM ALL

    @staticmethod
    def flags():
        # Define the argument parser
        parser = argparse.ArgumentParser(
            description="Logicytics, The most powerful tool for system data analysis."
        )
        # Define flags
        parser.add_argument(
            "--basic", action="store_true", help="Runs Logicytics default"
        )
        parser.add_argument(
            "--minimal", action="store_true", help="Run Logicytics in minimal mode. Just bare essential scraping"
        )
        parser.add_argument(
            "--unzip-extra",
            action="store_true",
            help="Unzip the extra directory zip file - Use on your own device only -.",
        )
        parser.add_argument(
            "--backup",
            action="store_true",
            help="Backup Logicytics files to the ACCESS/BACKUPS directory - Use on your own device only -.",
        )
        parser.add_argument(
            "--restore",
            action="store_true",
            help="Restore Logicytics files from the ACCESS/BACKUPS directory - Use on your own device only -.",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help="Update Logicytics from GitHub - Use on your own device only -.",
        )
        parser.add_argument(
            "--extra",
            action="store_true",
            help="Open the extra directory for more tools.",
        )
        parser.add_argument(
            "--dev",
            action="store_true",
            help="Run Logicytics developer mod, this is only for people who want to register their contributions properly. - Use on your own device only -.",
        )
        parser.add_argument(
            "--exe",
            action="store_true",
            help="Run Logicytics using its precompiled exe's, These may be outdated and not the best, use only if the device doesnt have python installed.",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Runs the Debugger, Will check for any issues, warning etc, useful for debugging and issue reporting",
        )
        parser.add_argument(
            "--modded",
            action="store_true",
            help="Runs the normal Logicytics, as well as any file in the MODS directory, Useful for custom scripts",
        )
        parser.add_argument(
            "--speedy",
            action="store_true",
            help="Runs Logicytics Speedy mode, where it runs in parallel only fast files",
        )
        parser.add_argument(
            "--webhook",
            action="store_true",
            help="Special Flag that will send zip file via webhook",
        )
        parser.add_argument(
            "--reboot",
            action="store_true",
            help="Special Flag that will reboot the device afterward",
        )
        parser.add_argument(
            "--shutdown",
            action="store_true",
            help="Special Flag that will shutdown the device afterward",
        )
        args = parser.parse_args()
        special_flag_used = False

        empty_check = (
            str(args)
            .removeprefix("Namespace(")
            .removesuffix(")")
            .replace("=", " = ")
            .replace(",", " ")
            .split(" ")
        )
        if "True" not in empty_check:
            parser.print_help()
            exit(1)

        # Check for exclusivity rules
        if args.reboot or args.shutdown or args.webhook:
            if not (
                args.basic
                or args.speedy
                or args.modded
                or args.silent
                or args.minimal
                or args.exe
            ):
                log.critical(
                    "--reboot and --shutdown and --webhook flags require at least one of the following flags: --basic, --speedy, --modded, --minimal, --exe."
                )
                exit(1)
            else:
                special_flag_used = True

        if not special_flag_used:
            # Ensure only one flag is used
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 1:
                log.critical("Only one flag is allowed.")
                exit(1)
        else:
            # Ensure only 2 flags is used
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 2:
                log.critical(
                    "Only one flag is allowed with the --reboot and --shutdown and --webhook flags."
                )
                exit(1)

        # Set flags to True or False based on whether they were used
        flags = {key: getattr(args, key) for key in vars(args)}

        # Initialize an empty list to store the keys with values set to True
        true_keys = []

        # Iterate through the flags dictionary
        for key, value in flags.items():
            # Check if the value is True and add the key to the list
            if value:
                true_keys.append(key)
                # Stop after adding two keys
                if len(true_keys) == 2:
                    break

        # Convert the list to a tuple and return it
        if len(tuple(true_keys)) < 3:
            return tuple(true_keys)
        else:
            log.critical(
                "Only one flag is allowed with the --reboot and --shutdown and --webhook flags."
            )
            exit(1)

    @staticmethod
    def read_config():
        try:
            with open("Libs/config.json", "r") as file:
                data = json.load(file)

                webhook_url = data.get("WEBHOOK_URL", "")
                debug = data.get("DEBUG", False)
                version = data.get("VERSION", "2.0.0")
                api_key = data.get("ipgeolocation.io API KEY", "")

                if not (
                    isinstance(webhook_url, str)
                    and isinstance(debug, bool)
                    and isinstance(version, str)
                    and isinstance(api_key, str)
                ):
                    log.critical("Invalid config.json format.")
                    exit(1)

                return webhook_url, debug, version, api_key
        except FileNotFoundError:
            log.critical("The config.json file is not found.")
            exit(1)


log = Log()
WEBHOOK, DEBUG, VERSION, API_KEY = Actions().read_config()
