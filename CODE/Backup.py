import shutil
from datetime import datetime
from local_libraries.Setups import *


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
        crash("OGE", "fun6", e, "error")
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
            crash("DE", "fun21", os.path.exists(source_dir), "error")
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
        crash("OGE", "fun21", e, "error")


create_backup()
