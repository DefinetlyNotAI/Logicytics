import getpass
import shutil
import zipfile

from local_libraries.Setups import *

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
        crash("DE", "fun10", os.path.exists(source_folder), "error")
        return

    # Create a ZipFile object
    with zipfile.ZipFile(destination_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
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
    data_dir = os.path.join(current_dir, "DATA")

    # Ensure the DATA directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # List all items in the current directory
    items = os.listdir(current_dir)

    # Filter items that are files with .txt, .file extensions or no extension
    target_files = [
        item
        for item in items
        if item.endswith(".txt") or item.endswith(".file") or not os.path.splitext(item)[1]
    ]

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
                    crash("PE", "fun37", e, "error")
                    continue  # Skip this iteration and move to the next item

                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred while processing {item}: {e}"
                    )
                    crash("OGE", "fun37", e, "error")
                    continue  # Skip this iteration and move to the next item
            else:
                logger.info(
                    f"Skipping {item} as it is not a file (it might be a directory)."
                )
    else:
        logger.warning(
            "No `.txt` files or files without extensions found in the current directory."
        )


process_files()
zip_data_folder()
logger.info("Finished Zipping the files")
