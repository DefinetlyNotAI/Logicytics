import shutil
import os
import subprocess
import colorlog
from datetime import datetime


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
    process = subprocess.Popen(r'powershell.exe -Command "& .\Crash_Reporter.py"', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
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
logger.setLevel(colorlog.DEBUG)  # Set the log level
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


def filter_zip_files(names):
    """
    Function to filter out zip files from being included in the backup.

    :param names: List of filenames in the source directory.
    :return: List of filenames to exclude (zip files).
    """
    try:
        return [name for name in names if name.endswith('.zip')]
    except Exception as e:
        logger.error(f"Error filtering zip files: {e}")
        crash("OGE", "fun69", e, "error")
        raise


def create_backup():
    """
    Function to create a backup of the CODE directory in the BACKUP directory.
    """
    try:
        # Get the absolute path of the parent directory
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Define the source directory (CODE) and the destination directory (BACKUP) in the parent directory
        source_dir = os.path.join(parent_dir, 'CODE')
        backup_dir = os.path.join(parent_dir, 'BACKUP')

        # Check if the source directory exists
        if not os.path.exists(source_dir):
            logger.error(f"Source directory does not exist: {source_dir}")
            crash("DE", "fun84", os.path.exists(source_dir), "error")
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        # Create the BACKUP directory if it doesn't already exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            logger.info(f"Created backup directory: {backup_dir}")

        # Get the current date and time as a string
        current_datetime_str = datetime.now().strftime('%d-%m-%Y')

        # Create the zip file name with the current date and time
        zip_name = f'{current_datetime_str}_backup'

        # Define the path for the zip file
        zip_file_path = os.path.join(backup_dir, zip_name)

        # Use shutil.make_archive to create a zip file of the CODE directory, ignoring zip files
        shutil.make_archive(base_name=os.path.join(backup_dir, zip_name), format='zip', root_dir=source_dir)

        logger.info(f"Backup created at {zip_file_path}.zip")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        crash("OGE", "fun84", e, "error")


create_backup()
