import ctypes
import os
from datetime import datetime
import colorlog
import subprocess
import argparse

# if __name__ == "__main__":
#     action, powerdown = Actions().flags()


# os.makedirs("../ACCESS/LOGS/", exist_ok=True)
# log = Log(filename="../ACCESS/LOGS/Logicytics.log", debug=True)


class Log:
    def __init__(
            self,
            filename="Server.log",
            err_filename=None,
            use_colorlog=True,
            debug=False,
            debug_color="cyan",
            info_color="green",
            warning_color="yellow",
            error_color="red",
            critical_color="red",
            colorlog_fmt_parameters="%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    ):
        """
        Initializes a new instance of the LOG class.

        The log class logs every interaction when called in both colorlog and in the log file

        Best to only modify filename, and DEBUG.

        Only if you are planning to use the dual-log parameter that allows you to both log unto the shell and the log file:
            IMPORTANT: This class requires colorlog to be installed and also uses it in the INFO level,
            To use the debug level, set DEBUG to True.

            If you are using colorlog, DO NOT INITIALIZE IT MANUALLY, USE THE LOG CLASS PARAMETER'S INSTEAD.
            Sorry for any inconvenience that may arise.

        Args:
            filename (str, optional): The name of the log file. Defaults to "Server.log".
            use_colorlog (bool, optional): Whether to use colorlog. Defaults to True.
            debug (bool, optional): Whether to use the debug level. Defaults to False (which uses the INFO level).
            debug_color (str, optional): The color of the debug level. Defaults to "cyan".
            info_color (str, optional): The color of the info level. Defaults to "green".
            warning_color (str, optional): The color of the warning level. Defaults to "yellow".
            error_color (str, optional): The color of the error level. Defaults to "red".
            critical_color (str, optional): The color of the critical level. Defaults to "red".
            colorlog_fmt_parameters (str, optional): The format of the log message. Defaults to "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s".

        Returns:
            None
        """
        self.color = use_colorlog
        if self.color:
            # Configure colorlog for logging messages with colors
            logger = colorlog.getLogger()
            if debug:
                logger.setLevel(
                    colorlog.DEBUG
                )  # Set the log level to DEBUG to capture all relevant logs
            else:
                logger.setLevel(
                    colorlog.INFO
                )  # Set the log level to INFO to capture all relevant logs
            handler = colorlog.StreamHandler()
            formatter = colorlog.ColoredFormatter(
                colorlog_fmt_parameters,
                datefmt=None,
                reset=True,
                log_colors={
                    "DEBUG": debug_color,
                    "INFO": info_color,
                    "WARNING": warning_color,
                    "ERROR": error_color,
                    "CRITICAL": critical_color,
                },
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)


        self.filename = str(filename)
        if err_filename is None:
            self.err_filename = self.filename
        else:
            self.err_filename = str(err_filename)
        if not os.path.exists(self.filename):
            self.__only("|" + "-" * 19 + "|" + "-" * 13 + "|" + "-" * 154 + "|")
            self.__only(
                "|     Timestamp     |  LOG Level  |"
                + " " * 71
                + "LOG Messages"
                + " " * 71
                + "|"
            )
        self.__only("|" + "-" * 19 + "|" + "-" * 13 + "|" + "-" * 154 + "|")

    @staticmethod
    def __timestamp() -> str:
        """
        Returns the current timestamp as a string in the format 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            str: The current timestamp.
        """
        now = datetime.now()
        time = f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
        return time

    def __only(self, message):
        """
        Logs a quick message to the log file.

        Args:
            message: The message to be logged.

        Returns:
            None
        """
        with open(self.filename, "a") as f:
            f.write(f"{str(message)}\n")

    @staticmethod
    def __pad_message(message):
        """
        Adds spaces to the end of a message until its length is exactly 153 characters.

        Parameters:
        - message (str): The input message string.

        Returns:
        - str: The padded message with a length of exactly 153 characters.
        """
        # Calculate the number of spaces needed
        num_spaces = 153 - len(message)

        if num_spaces > 0:
            # If the message is shorter than 153 characters, add spaces to the end
            padded_message = message + " " * num_spaces
        else:
            # If the message is already longer than 153 characters, truncate it to the first 153 characters
            padded_message = message[:150]
            padded_message += "..."

        padded_message += "|"
        return padded_message

    def info(self, message):
        """
        Logs an informational message to the log file.

        Args:
            message: The message to be logged.

        Returns:
            None
        """
        if self.color:
            colorlog.info(message)
        with open(self.filename, "a") as f:
            f.write(
                f"[{self.__timestamp()}] > INFO:     | {self.__pad_message(str(message))}\n"
            )

    def warning(self, message):
        """
        Logs a warning message to the log file.

        Args:
            message: The warning message to be logged.

        Returns:
            None
        """
        if self.color:
            colorlog.warning(message)
        with open(self.filename, "a") as f:
            f.write(
                f"[{self.__timestamp()}] > WARNING:  | {self.__pad_message(str(message))}\n"
            )

    def error(self, message):
        """
        Logs an error message to the log file.

        Args:
            message: The error message to be logged.

        Returns:
            None
        """
        if self.color:
            colorlog.error(message)
        with open(self.err_filename, "a") as f:
            f.write(
                f"[{self.__timestamp()}] > ERROR:    | {self.__pad_message(str(message))}\n"
            )

    def critical(self, message):
        """
        Writes a critical message to the log file.

        Args:
            message: The critical message to be logged.

        Returns:
            None
        """
        if self.color:
            colorlog.critical(message)
        with open(self.err_filename, "a") as f:
            f.write(
                f"[{self.__timestamp()}] > CRITICAL: | {self.__pad_message(str(message))}\n"
            )

class Checks:
    def __init__(self):
        self.Actions = Actions()

    @staticmethod
    def is_admin() -> bool:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    def using_uac(self) -> bool:
        value = self.Actions.run_command("powershell (Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System).EnableLUA")
        return int(value.strip("\n")) == 1

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
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--modded", action="store_true")
        parser.add_argument("--speedy", action="store_true")
        parser.add_argument("--basic", action="store_true")

        args = parser.parse_args()
        skip = False

        empty_check = str(args).removeprefix("Namespace(").removesuffix(")").replace("=", " = ").replace(",", " ").split(" ")
        if "True" not in empty_check:
            parser.print_help()
            exit(1)

        # Check for exclusivity rules
        if args.reboot or args.shutdown:
            if not (args.basic or args.speedy or args.modded or args.silent or args.minimal or args.exe):
                print("Error: --reboot and --shutdown flags require at least one of the following flags: --basic, --speedy, --modded, --silent, --minimal, --exe.")
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
                print("Error: Only one flag is allowed with the --reboot and --shutdown flags.")
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
        return tuple(true_keys)
