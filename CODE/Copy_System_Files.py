import getpass
import os
import shutil
import subprocess
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

USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME

paths_and_name = [
    # Your list of paths and names here...
]


def copy_and_rename_files(paths_and_name):
    for file_path, file_name in zip(paths_and_name[::2], paths_and_name[1::2]):
        try:
            file_path = os.path.expandvars(file_path)
            if not os.path.exists(file_path):
                logger.error(f"The file {file_path} does not exist.")
                continue

            shutil.copy2(file_path, os.getcwd())
            new_file_name = f"{USER_NAME}_{file_name}"
            new_file_path = os.path.join(os.getcwd(), new_file_name)
            if os.path.exists(new_file_path):
                os.remove(new_file_path)  # Delete the existing file
            os.rename(os.path.join(os.getcwd(), os.path.basename(file_path)), new_file_path)
            logger.info(f"Copied and renamed file to {new_file_name}")
        except FileNotFoundError:
            logger.error(f"The file at path {file_path} was not found.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


def execute_tree_batch_file():
    batch_file_name = "Tree_Command.bat"
    if os.path.exists(batch_file_name):
        subprocess.run([batch_file_name], check=True)
        logger.info(f"{batch_file_name} has been executed successfully.")
    else:
        logger.error(f"{batch_file_name} not found in the current working directory.")


execute_tree_batch_file()
copy_and_rename_files(paths_and_name)
