import configparser
import os


def __config_data() -> tuple[str, str, list[str], bool, str]:
    """
    Retrieves configuration data from the 'config.ini' file.

    If the configuration file is not found in any of these locations,
    the program exits with an error message.

    Returns:
        tuple[str, str, list[str], bool]: A tuple containing:
            - Log level (str): Either "DEBUG" or "INFO"
            - Version (str): System version from configuration
            - Files (list[str]): List of files specified in configuration
            - Delete old logs (bool): Flag indicating whether to delete old log files
            - CONFIG itself

    Raises:
        SystemExit: If the 'config.ini' file cannot be found in any of the attempted locations
    """

    def _config_path() -> str:
        configs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")

        if os.path.exists(configs_path):
            return configs_path
        else:
            print("The config.ini file is not found in the expected location.")
            exit(1)

    config = configparser.ConfigParser()
    path = _config_path()
    config.read(path)

    log_using_debug = config.getboolean("Settings", "log_using_debug")
    delete_old_logs = config.getboolean("Settings", "delete_old_logs")
    version = config.get("System Settings", "version")
    files = config.get("System Settings", "files").split(", ")

    log_using_debug = "DEBUG" if log_using_debug else "INFO"

    return log_using_debug, version, files, delete_old_logs, config


# Check if the script is being run directly, if not, set up the library
if __name__ == '__main__':
    exit("This is a library, Please import rather than directly run.")
DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS, CONFIG = __config_data()
