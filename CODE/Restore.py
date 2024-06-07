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


def unzip_backup():
    """Unzips the selected backup file into the CODE directory.
    This function navigates to the parent directory and then to the BACKUP directory,
    lists all.zip files in the BACKUP directory, allows the user to select one,
    and then unzips the selected file into the CODE directory, replacing files if required.
    """
    try:
        # Step 1 & 2: Navigate to the parent directory and then to the BACKUP directory
        current_dir = os.getcwd()  # Get the current working directory
        parent_dir = os.path.dirname(current_dir)  # Move up to the parent directory
        backup_dir_path = os.path.join(parent_dir, 'BACKUP')  # Path to the BACKUP directory

        if not os.path.exists(backup_dir_path):
            raise FileNotFoundError(f"Backup directory '{backup_dir_path}' not found.")

        # Step 3: List all.zip files in the BACKUP directory and prompt the user to select one
        os.chdir(backup_dir_path)  # Change the current working directory to BACKUP
        zip_files = [file for file in os.listdir('.') if file.endswith('.zip')]

        if not zip_files:
            raise FileNotFoundError("No.zip files found in the backup directory.")

        print("Select the backup to unzip:")
        for index, zip_file in enumerate(zip_files, start=1):
            print(f"{index}. {zip_file}")

        selection = int(input("Enter the number of your selection: ")) - 1

        if selection < 0 or selection >= len(zip_files):
            raise ValueError("Invalid selection. Please enter a valid number.")

        zip_file_name = zip_files[selection]  # User-selected.zip file

        # Step 4: Move back to the parent directory
        os.chdir(parent_dir)

        # Step 5: Unzip the contents into the CODE directory, replacing files if required
        code_dir_path = os.path.join(parent_dir, 'CODE')  # Path to the CODE directory
        if not os.path.exists(code_dir_path):
            os.makedirs(code_dir_path)  # Create the CODE directory if it doesn't exist

        with zipfile.ZipFile(os.path.join(backup_dir_path, zip_file_name), 'r') as zip_ref:
            zip_ref.extractall(
                code_dir_path)  # Extracts all the files into CODE directory, replacing them if they exist

        logger.info(f"Unzipped {zip_file_name} into {code_dir_path}")

    except Exception as e:
        logger.error(f"An error occurred while unzipping the backup: {str(e)}")


# Call the function to start unzipping the backup
unzip_backup()
