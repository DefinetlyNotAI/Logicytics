import os
import shutil
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
            total_size += os.path.getsize(fp)
    return total_size


def copy_folders(source_paths, destination_path):
    """Copy folders to a specified destination."""
    for source_path in source_paths:
        try:
            shutil.copytree(str(source_path), os.path.join(str(destination_path), os.path.basename(str(source_path))))
            logger.info(f"Folder '{os.path.basename(source_path)}' copied successfully.")
        except Exception as e:
            logger.error(f"Failed to copy folder '{os.path.basename(source_path)}': {e}")


def main():
    # Get the current user's username
    username = os.getlogin()

    # Define the source folders using the current user's username
    source_folders = [
        f"C:/Users/{username}/Music",
        f"C:/Users/{username}/Pictures",
        f"C:/Users/{username}/Videos"
    ]

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

    # Proceed with copying the folders without user confirmation
    copy_folders(source_folders, destination_folder)
    logger.info("Folders copied successfully.")


if __name__ == "__main__":
    main()
