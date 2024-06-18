import os
import shutil
import colorlog
import subprocess


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


def format_size(size_bytes):
    """Format size into KB, MB, GB"""
    if size_bytes >= 1073741824:  # 1 GB
        return f"{size_bytes / 1073741824:.2f} GB"
    elif size_bytes >= 1048576:  # 1 MB
        return f"{size_bytes / 1048576:.2f} MB"
    elif size_bytes >= 1024:  # 1 KB
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bytes"


def estimate_folder_size(folder_path):
    """Estimate the size of a folder."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(str(folder_path)):
        for f in filenames:
            fp = os.path.join(str(dirpath), f)
            if os.path.isfile(fp):  # Check if the path is a file
                total_size += os.path.getsize(fp)
    return total_size


def copy_folders(source_paths, destination_path):
    """Copy folders to a specified destination."""
    for source_path in source_paths:
        try:
            shutil.copytree(str(source_path), os.path.join(str(destination_path), os.path.basename(str(source_path))))
            logger.info(f"Folder '{os.path.basename(source_path)}' copied successfully.")
        except PermissionError as e:
            logger.error(f"Permission denied while trying to copy folder '{os.path.basename(source_path)}': {e}")
            crash("PE", "fun91", e, "error")
        except OSError as e:
            logger.error(f"An error occurred while trying to copy folder '{os.path.basename(source_path)}': {e}")
            crash("OSE", "fun91", e, "error")
        except Exception as e:
            logger.error(
                f"Unexpected error occurred while trying to copy folder '{os.path.basename(source_path)}': {e}")
            crash("OGE", "fun91", e, "error")


def main():
    """
    Main function to copy media folders from the user's profile to a destination folder.
    """
    # Get the current user's username
    username = os.getlogin()

    # Define the source folders using the current user's username
    source_folders = [
        f"C:/Users/{username}/Music",
        f"C:/Users/{username}/Pictures",
        f"C:/Users/{username}/Videos"
    ]

    # Check if the source folders exist
    for folder in source_folders:
        if not os.path.exists(folder):
            logger.error(f"Source folder does not exist: {folder}")
            crash("FNF", "fun109", os.path.exists(folder), "error")
            continue

    # Get the script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the destination folder as a DATA folder within the script's directory
    destination_folder = os.path.join(script_dir, "DATA")

    # Create the DATA folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        logger.info("Created DATA folder.")

    # Estimate the sizes of the source folders
    estimated_sizes = {}
    for folder in source_folders:
        if os.path.exists(folder):
            estimated_size = estimate_folder_size(folder)
            formatted_size = format_size(estimated_size)
            estimated_sizes[folder] = formatted_size
            logger.info(f"Estimated size of '{folder}': {formatted_size}")
        else:
            logger.error(f"ERROR: Folder not found: {folder}")
            crash("FNF", "fun109", os.path.exists(folder), "error")

    # Proceed with copying the folders without user confirmation
    copy_folders(source_folders, destination_folder)
    logger.info("Folders copied successfully.")


if __name__ == "__main__":
    main()
