import os
import sys
import platform
from datetime import datetime
import requests
from __lib_actions import *
from __lib_log import Log


class SystemInfo:
    def __init__(self):
        self.device_model = platform.machine()
        self.python_version = sys.version.split()[0]
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.is_vm = os.environ.get('VIRTUAL_ENV') is not None
        self.is_admin = os.getpid() == 0
        self.execution_policy = os.popen('powershell Get-ExecutionPolicy').read().strip()
        self.os_name = platform.system()
        self.os_version = platform.release()
        self.manufacturer = platform.processor()

    @staticmethod
    def get_date_time():
        now = datetime.now()
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def is_admin(self):
        return os.environ.get('PROCESSOR_ARCH') == 'x86_64'

    @is_admin.setter
    def is_admin(self, value):
        self._is_admin = value


class JSON:
    @staticmethod
    def check_current_files(directory):
        file = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(('.py', '.exe', '.ps1', '.bat')):
                    files_path = os.path.join(root, filename)
                    file.append(files_path.removeprefix('.\\'))
        return file

    @staticmethod
    def update_json_file(filename, new_array):
        with open(filename, 'r+') as f:
            data = json.load(f)
            data['CURRENT_FILES'] = new_array
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    @staticmethod
    def get_json_data(URL):
        response = requests.get(URL)
        return response.json()


def debug():
    # Set required variables
    info = SystemInfo()
    log = Log(debug=True, filename="../ACCESS/LOGS/DEBUG/Logicytics_Debug.log")
    JSON.update_json_file('config.json', JSON.check_current_files('.'))
    data = JSON.get_json_data("https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.json")
    Nx1, Nx2, N_VERSION, Nx3, N_CURRENT_FILES = Actions().read_config()
    extra_in_config = set(N_CURRENT_FILES).difference(set(data['CURRENT_FILES']))
    missing_in_config = set(data['CURRENT_FILES']).difference(set(N_CURRENT_FILES))
    diff = set(data['CURRENT_FILES']).symmetric_difference(set(N_CURRENT_FILES))

    # Create output
    log.info(info.device_model + " " + info.os_name + " " + info.os_version + " " + info.manufacturer)
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
    if data['VERSION'] == N_VERSION:
        log.info(f"Up to date: {VERSION}")
    elif data['VERSION'] >= N_VERSION:
        log.warning(f"Not up to date: {VERSION}")
    else:
        log.warning(f"Modified: {VERSION}")
