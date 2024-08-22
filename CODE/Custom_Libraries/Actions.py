import subprocess
import argparse
import json

class Actions:
    @staticmethod
    def run_command(command):
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process.stdout

    @staticmethod
    def flags():
        # Define the argument parser
        parser = argparse.ArgumentParser(description="Logicytics, The most powerful tool for system data analysis.")

        # Define flags
        parser.add_argument("--minimal", action="store_true")
        parser.add_argument("--unzip-extra", action="store_true")
        parser.add_argument("--backup", action="store_true")
        parser.add_argument("--restore", action="store_true")
        parser.add_argument("--update", action="store_true")
        parser.add_argument("--extra", action="store_true")
        parser.add_argument("--dev", action="store_true")
        parser.add_argument("--exe", action="store_true")
        parser.add_argument("--silent", action="store_true")
        parser.add_argument("--reboot", action="store_true")
        parser.add_argument("--shutdown", action="store_true")
        parser.add_argument("--DEBUG", action="store_true")
        parser.add_argument("--modded", action="store_true")
        parser.add_argument("--speedy", action="store_true")
        parser.add_argument("--basic", action="store_true")
        parser.add_argument("--webhook", action="store_true")

        args = parser.parse_args()
        skip = False

        empty_check = str(args).removeprefix("Namespace(").removesuffix(")").replace("=", " = ").replace(",", " ").split(" ")
        if "True" not in empty_check:
            parser.print_help()
            exit(1)

        # Check for exclusivity rules
        if args.reboot or args.shutdown or args.webhook:
            if not (args.basic or args.speedy or args.modded or args.silent or args.minimal or args.exe):
                print("Error: --reboot and --shutdown and --webhook flags require at least one of the following flags: --basic, --speedy, --modded, --silent, --minimal, --exe.")
                exit(1)
            else:
                skip = True

        if not skip:
            # Ensure only one flag is used
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 1:
                print("Error: Only one flag is allowed.")
                exit(1)
        else:
            # Ensure only 2 flags is used
            used_flags = [flag for flag in vars(args) if getattr(args, flag)]
            if len(used_flags) > 2:
                print("Error: Only one flag is allowed with the --reboot and --shutdown and --webhook flags.")
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
            print("Error: Only one flag is allowed with the --reboot and --shutdown and --webhook flags.")
            exit(1)

    @staticmethod
    def read_config():
        try:
            with open('../SYSTEM/config.json', 'r') as file:
                data = json.load(file)

                webhook_url = data.get('WEBHOOK_URL', '')
                debug = data.get('DEBUG', False)
                version = data.get('VERSION', '2.0.0')
                files = data.get('FILES', [])

                if not (all(isinstance(file, str) for file in files) and isinstance(webhook_url, str) and isinstance(debug, bool) and isinstance(version, str)):
                    print("Invalid config.json format.")
                    exit(1)

                return webhook_url, debug, version, files
        except FileNotFoundError:
            print("config.json not found.")
            exit(1)
