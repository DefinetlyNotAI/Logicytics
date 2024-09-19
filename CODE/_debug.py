from __future__ import annotations

import requests
from __lib_class import *


# TODO - Add a debug to do the following to the project
# [2024-09-18 12:50:00] > INFO:     | AMD64 Windows 10 Intel64 Family 6 Model 186 Stepping 2, GenuineIntel                                                                                   |
# [2024-09-18 12:50:00] > INFO:     | 3.11.9                                                                                                                                                 |
# [2024-09-18 12:50:00] > INFO:     | C:\Users\Hp\Desktop\Repositories\Logicytics\CODE                                                                                                       |
# [2024-09-18 12:50:00] > INFO:     | Is VM: False                                                                                                                                           |
# [2024-09-18 12:50:00] > INFO:     | Running as admin: False                                                                                                                                |
# [2024-09-18 12:50:00] > INFO:     | Execution policy: Unrestricted                                                                                                                         |
# [2024-09-18 12:50:00] > WARNING:  | Extra in your config.json: {'property_scraper.ps1', 'window_feature_miner.ps1', '_debug.py', '_hide_my_tracks.py', 'tree.bat', '__lib_class.py', 'ss...|
# [2024-09-18 12:50:00] > WARNING:  | SysInternal Binaries Not Found: Zipped

def debug():
    pass


class HealthCheck:
    def get_config_data(self) -> bool | tuple[tuple[str, str, str], tuple[str, str, str]]:
        try:
            url = "https://raw.githubusercontent.com/DefinetlyNotAI/Logicytics/main/CODE/config.json"
            config = json.loads(requests.get(url).text)
        except requests.exceptions.ConnectionError:
            log.warning("No connection found")
            return False
        version_check = self.__compare_versions(VERSION, config['VERSION'])
        file_check = self.__check_files(CURRENT_FILES, config['CURRENT_FILES'])

        return version_check, file_check

    @staticmethod
    def __compare_versions(local_version, remote_version) -> tuple[str, str, str]:
        if local_version == remote_version:
            return "Version is up to date.", f"Your Version: {local_version}", "INFO"
        elif local_version > remote_version:
            return "Version is ahead of the repository.", f"Your Version: {local_version}, Repository Version: {remote_version}", "WARNING"
        else:
            return "Version is behind the repository.", f"Your Version: {local_version}, Repository Version: {remote_version}", "ERROR"

    @staticmethod
    def __check_files(local_files, remote_files) -> tuple[str, str, str]:
        missing_files = set(remote_files) - set(local_files)
        if not missing_files:
            return "All files are present.", f"Your files: {local_files} contain all the files in the repository.", "INFO"
        else:
            return "You have missing files.", f"You are missing the following files: {missing_files}", "ERROR"


if HealthCheck().get_config_data():
    version_tuple, file_tuple = HealthCheck().get_config_data()
    log_funcs.get(version_tuple[3], log.debug)("\n".join(version_tuple[1]).replace('\n', ''))
    log_funcs.get(file_tuple[3], log.debug)("\n".join(file_tuple[1]).replace('\n', ''))
