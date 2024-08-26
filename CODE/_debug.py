import os
import sys
import platform
from datetime import datetime
import requests
from __lib_actions import *
from __lib_log import Log


class SystemInfo:
    def __init__(self):
        """
        Initializes a new instance of the SystemInfo class.

        This constructor sets various system properties, including:
        - device_model: The machine type of the device.
        - python_version: The version of the Python interpreter.
        - current_path: The absolute path of the current file.
        - is_vm: A boolean indicating whether the environment is a virtual environment.
        - is_admin: A boolean indicating whether the process is running with administrative privileges.
        - execution_policy: The execution policy of the system.
        - os_name: The name of the operating system.
        - os_version: The version of the operating system.
        - manufacturer: The processor manufacturer.

        Parameters:
        None

        Returns:
        None
        """
        self.device_model = platform.machine()
        self.python_version = sys.version.split()[0]
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.is_vm = os.environ.get("VIRTUAL_ENV") is not None
        self.is_admin = os.getpid() == 0
        self.execution_policy = (
            os.popen("powershell Get-ExecutionPolicy").read().strip()
        )
        self.os_name = platform.system()
        self.os_version = platform.release()
        self.manufacturer = platform.processor()

    @staticmethod
    def get_date_time() -> str:
        """
        Get the current date and time in the format 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            str: The current date and time in the format 'YYYY-MM-DD HH:MM:SS'.
        """
        now = datetime.now()
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def is_admin(self) -> bool:
        """
        Checks if the current process is running with administrative privileges.

        Returns:
            bool: True if the process is running as an administrator, False otherwise.
        """
        return os.environ.get("PROCESSOR_ARCH") == "x86_64"

    @is_admin.setter
    def is_admin(self, value: bool):
        """
        Sets the value of the is_admin property.

        Parameters:
        value (bool): The new value for the is_admin property.

        Returns:
        None
        """
        self._is_admin = value


class JSON:
    @staticmethod
    def check_current_files(directory: str) -> list:
        """
        Checks the specified directory and its subdirectories for files with extensions '.py', '.exe', '.ps1', or '.bat'.

        Parameters:
            directory (str): The path to the directory to search.

        Returns:
            list: A list of file paths with the specified extensions. The paths are relative to the directory and do not include the directory prefix.
        """
        file = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith((".py", ".exe", ".ps1", ".bat")):
                    files_path = os.path.join(root, filename)
                    file.append(files_path.removeprefix(".\\"))
        return file

    @staticmethod
    def update_json_file(filename: str, new_array: list):
        """
        Updates a JSON file with a new array of current files.

        Parameters:
            filename (str): The path to the JSON file to update.
            new_array (list): The new array of current files to write to the JSON file.

        Returns:
            None
        """
        with open(filename, "r+") as f:
            data = json.load(f)
            data["CURRENT_FILES"] = new_array
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    @staticmethod
    def get_json_data(URL: str) -> dict:
        """
        Retrieves data from a specified URL and returns it as a dictionary.

        Parameters:
        URL (str): The URL to retrieve data from.

        Returns:
        dict: A dictionary containing the retrieved data.
        """
        response = requests.get(URL)
        return response.json()


def debug():
    # Set required variables
    info = SystemInfo()
    log = Log(debug=True, filename="../ACCESS/LOGS/DEBUG/Logicytics_Debug.log")
    JSON.update_json_file("config.json", JSON.check_current_files("."))
    data = JSON.get_json_data(
        "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.json"
    )
    Nx1, Nx2, N_VERSION, Nx3, N_CURRENT_FILES = Actions().read_config()
    extra_in_config = set(N_CURRENT_FILES).difference(set(data["CURRENT_FILES"]))
    missing_in_config = set(data["CURRENT_FILES"]).difference(set(N_CURRENT_FILES))
    diff = set(data["CURRENT_FILES"]).symmetric_difference(set(N_CURRENT_FILES))

    # Create output
    log.info(
        info.device_model
        + " "
        + info.os_name
        + " "
        + info.os_version
        + " "
        + info.manufacturer
    )
    log.info(info.python_version)
    log.info(info.current_path)
    log.info(f"Is VM: {info.is_vm}")
    log.info(f"Running as admin: {info.is_admin}")
    log.info(f"Execution policy: {info.execution_policy}")
    log.info(f"Date and time: {info.get_date_time()}")
    if diff != set():
        log.warning(f"Differences: {diff}")
    if missing_in_config != set():
        log.error(f"Missing in your config.json: {missing_in_config}")
        log.critical("Corruption Found", "_d", "C", "BA")
    if extra_in_config != set():
        log.warning(f"Extra in your config.json: {extra_in_config}")
    if len(diff) == 0 and len(missing_in_config) == 0 and len(extra_in_config) == 0:
        log.info("Files are up-to date, No differences found")
    if data["VERSION"] == N_VERSION:
        log.info(f"Up to date: {VERSION}")
    elif data["VERSION"] >= N_VERSION:
        log.warning(f"Not up to date: {VERSION}")
    else:
        log.warning(f"Modified: {VERSION}")
