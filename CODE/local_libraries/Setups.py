# Logging and crash reporting occur in this local library

import os
import subprocess
import colorlog


def print_colored(text, color):
    """
    Prints the given text in the specified color.

    Parameters:
        text (str): The text to print.
        color (str): The color code (e.g., 'red', 'green', etc.).

    Returns:
        None

    Raises:
        ValueError: If the color name is invalid.

    Example:
        print_colored("Hello, world!", "red")
        # Output: Hello, world! (in red color)
    """
    reset = "\033[0m"
    color_codes = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
    }
    if color.lower() in color_codes:
        print(color_codes[color.lower()] + text + reset)


def crash(error_id, function_no, error_content, type):
    """
    Ensure error_id and function_no are strings
    Prepare the data to write to the temporary files
    Write the name of the placeholder script to the temporary file
    Write the error message to the temporary file
    Write the name of the placeholder function to the temporary file
    Write the name of the placeholder language to the temporary file
    Write the name of the placeholder crash to the temporary file
    Write the type to the temporary file
    Open Crash_Reporter.py in a new shell window
    """
    # Ensure error_id and function_no are strings
    error_id = str(error_id)
    function_no = str(function_no)

    # Prepare the data to write to the temporary files
    script_name = os.path.basename(__file__)
    language = os.path.splitext(__file__)[1][1:]  # Extracting the language part

    # Write the name of the placeholder script to the temporary file
    with open("flag.temp", "w") as f:
        f.write(script_name)

    # Write the error message to the temporary file
    with open("error.temp", "w") as f:
        f.write(error_id)

    # Write the name of the placeholder function to the temporary file
    with open("function.temp", "w") as f:
        f.write(function_no)

    # Write the name of the placeholder language to the temporary file
    with open("language.temp", "w") as f:
        f.write(language)

    # Write the name of the placeholder crash to the temporary file
    with open("error_data.temp", "w") as f:
        f.write(error_content)

    with open("type.temp", "w") as f:
        f.write(type)

    # Open Crash_Reporter.py in a new shell window
    # Note: This command works for Command Prompt.
    # Adjust according to your needs.
    process = subprocess.Popen(
        r'powershell.exe -Command "& .\Crash_Reporter.py"',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(process.stdout.readline, b""):
        decoded_line = line.decode("utf-8").strip()
        print(decoded_line)
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()
    # Decode the output from bytes to string
    stdout = stdout.decode("utf-8") if stdout else ""
    print(stdout)


# Configure colorlog for logging messages with colors
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level to INFO to capture all relevant logs

handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
handler.setFormatter(formatter)
logger.addHandler(handler)
