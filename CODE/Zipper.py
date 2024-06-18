import getpass
import os
import shutil
import subprocess
import zipfile
import colorlog


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
USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME


def zip_data_folder():
    """
    This function zips the contents of the 'DATA' folder into a zip file named after the user.
    """
    # Define the source folder and the destination zip file
    source_folder = "DATA"
    destination_zip = f"{USER_NAME}_data.zip"

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        logger.error(f"The folder {source_folder} does not exist.")
        crash("DE", "fun72", os.path.exists(source_folder), "error")
        return

    # Create a ZipFile object
    with zipfile.ZipFile(destination_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over all the files in the source folder
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Add the file to the zip
                zipf.write(file_path, os.path.relpath(file_path, source_folder))

    logger.info(f"Folder {source_folder} has been zipped into {destination_zip}.")


def process_files():
    """
    This function processes files in the current directory by moving text files,
    .file files, or files with no extension to the 'DATA' directory.
    """
    # Define the current working directory and the DATA directory
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'DATA')

    # Ensure the DATA directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # List all items in the current directory
    items = os.listdir(current_dir)

    # Filter items that are files with .txt, .file extensions or no extension
    target_files = [item for item in items if
                    item.endswith('.txt') or item.endswith('.file') or not os.path.splitext(item)[1]]

    if target_files:
        logger.info(f"Found {len(target_files)} files to process.")
        for item in target_files:
            # Construct the full path to the item
            item_path = os.path.join(current_dir, item)

            # Check if the item is a file before attempting to copy
            if os.path.isfile(item_path):
                try:
                    # Copy the file to the DATA directory
                    shutil.copy(item_path, data_dir)

                    # Attempt to delete the original file
                    os.remove(item_path)
                    logger.info(f"Processed {item}, copied to {data_dir} and deleted.")

                except PermissionError as e:
                    logger.error(f"Failed to delete {item}: {e}")
                    crash("PE", "fun99", e, "error")
                    continue  # Skip this iteration and move to the next item

                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {item}: {e}")
                    crash("OGE", "fun99", e, "error")
                    continue  # Skip this iteration and move to the next item
            else:
                logger.info(f"Skipping {item} as it is not a file (it might be a directory).")
    else:
        logger.warning("No `.txt` files or files without extensions found in the current directory.")


process_files()
zip_data_folder()
logger.info("Finished Zipping the files")
