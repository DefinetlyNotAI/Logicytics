from datetime import datetime
import getpass
import os
import shutil
import time
import zipfile

USER_NAME = getpass.getuser()
DESTINATION_PREFIX = "DATA\\" + USER_NAME


def zip_data_folder():
    # Define the source folder and the destination zip file
    source_folder = "DATA"
    destination_zip = f"{USER_NAME}_data.zip"

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"ERROR: The folder {source_folder} does not exist.")
        print()
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

    print(f"INFO: Folder {source_folder} has been zipped into {destination_zip}.")
    print()


def process_files():
    # Define the current working directory and the DATA directory
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'DATA')

    # Ensure the DATA directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # List all items in the current directory
    items = os.listdir(current_dir)

    # Filter items that are files with.txt,.file extensions or no extension
    target_files = [item for item in items if
                    item.endswith('.txt') or item.endswith('.file') or not os.path.splitext(item)[1]]

    if target_files:
        print(f"INFO: Found {len(target_files)} files to process.")
        print()
        for item in target_files:
            # Construct the full path to the item
            item_path = os.path.join(current_dir, item)

            # Check if the item is a file before attempting to copy
            if os.path.isfile(item_path):
                # Copy the file to the DATA directory
                shutil.copy(item_path, data_dir)

                # Delete the original file
                os.remove(item_path)

                print(f"INFO: Processed {item}, copied to {data_dir} and deleted.")
                print()
            else:
                print(f"INFO: Skipping {item} as it is not a file (it might be a directory).")
                print()
    else:
        print("WARNING: No.txt,.file files or files without extensions found in the current directory.")
        print()
        

def empty_data_folder():
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the folder name to search for
    folder_name = "DATA"

    # Construct the path to the folder
    folder_path = os.path.join(current_dir, folder_name)

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is a directory
        if os.path.isdir(folder_path):
            # List all files and directories in the folder
            for item in os.listdir(folder_path):
                # Construct the full path to the item
                item_path = os.path.join(folder_path, item)

                # Check if the item is a file or directory
                if os.path.isfile(item_path):
                    # Remove the file
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    # Remove the directory and its contents
                    shutil.rmtree(item_path)

    else:
        print(f"ERROR: The folder '{folder_name}' does not exist in the current working directory.")


def get_current_datetime():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_now
    

process_files()
time.sleep(6)
zip_data_folder()
print("INFO: Finished, Closing in 3 seconds...")
print()
time.sleep(6)
empty_data_folder()
current_datetime = get_current_datetime()
print("SYSTEM: Project Complete: ", current_datetime)
print()
