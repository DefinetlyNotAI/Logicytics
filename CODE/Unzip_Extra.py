import os
import zipfile
import colorlog

# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.DEBUG)  # Set the log level

# Create a StreamHandler for colorlog
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


def unzip_extra():
    """Automatically unzips all.zip files in the BACKUP directory into the CODE directory."""
    try:
        # Step 1 & 2: Navigate to the parent directory and then to the BACKUP directory
        current_dir = os.getcwd()  # Get the current working directory
        parent_dir = os.path.dirname(current_dir)  # Move up to the parent directory
        extra_dir_path = os.path.join(parent_dir, 'EXTRA')  # Path to the EXTRA directory

        # Check if the EXTRA directory exists
        if not os.path.exists(extra_dir_path):
            raise FileNotFoundError(f"Directory '{extra_dir_path}' not found.")

        # Step 3: List all .zip files in the EXTRA directory and unzip them automatically
        os.chdir(extra_dir_path)  # Change the current working directory to EXTRA
        zip_files = [file for file in os.listdir('.') if file.endswith('.zip')]

        if not zip_files:
            raise FileNotFoundError("No .zip files found in the backup directory.")

        # Step 4: Move back to the parent directory
        os.chdir(parent_dir)

        # Step 5: Unzip the contents into the EXTRA directory, replacing files if required
        if not os.path.exists(extra_dir_path):
            os.makedirs(extra_dir_path)  # Create the EXTRA directory if it doesn't exist

        for zip_file in zip_files:
            with zipfile.ZipFile(os.path.join(extra_dir_path, zip_file), 'r') as zip_ref:
                zip_ref.extractall(extra_dir_path)  # Extracts all the files into EXTRA directory, replacing them if they exist
            logger.info(f"Unzipped {zip_file} into {extra_dir_path}")

            # Delete the ZIP file after successful extraction
            os.remove(os.path.join(extra_dir_path, zip_file))
            logger.info(f"Deleted {zip_file}")

    except Exception as e:
        logger.error(f"An error occurred while unzipping the extra files: {str(e)}")


# Call the function to start unzipping the backups automatically
unzip_extra()
