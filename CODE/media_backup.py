import getpass
import os
import shutil
from datetime import datetime

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


class Media:
    """
    A class to handle media backup operations.
    """

    @staticmethod
    def __get_default_paths() -> list:
        """
        Returns the default paths for photos and videos based on the Windows username.
        
        This method retrieves the current Windows user's default media directories for photos and videos
        by using the current username and standard Windows file system paths.
        
        Returns:
            list: A list containing two paths:
                - First element: Default photo directory path
                - Second element: Default video directory path
        
        Notes:
            - Uses `getpass.getuser()` to dynamically retrieve the current Windows username
            - Expands the user path using `os.path.expanduser()` to handle potential path variations
            - Assumes standard Windows user directory structure
        """
        username = getpass.getuser()
        default_photo_path = os.path.expanduser(f"C:\\Users\\{username}\\Pictures")
        default_video_path = os.path.expanduser(f"C:\\Users\\{username}\\Videos")
        return [default_photo_path, default_video_path]

    @staticmethod
    def __ensure_backup_directory_exists(backup_directory: str):
        """
        Ensures the backup directory exists, creating it if necessary.
        
        Args:
            backup_directory (str): The full path to the directory where media files will be backed up.
        
        Raises:
            OSError: If the directory cannot be created due to permission issues or other system constraints.
        """
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

    @staticmethod
    def __collect_media_files(source_dirs: list) -> list:
        """
        Collects media files from specified source directories.
        
        Recursively searches through the provided source directories to find image and video files with extensions .jpg, .jpeg, .png, and .mp4.
        
        Args:
            source_dirs (list): List of directory paths to search for media files.
        
        Returns:
            list: Absolute file paths of all discovered media files, including those in subdirectories.
        
        Raises:
            OSError: If any of the source directories are inaccessible or cannot be traversed.
        """
        media_files = []
        for source_dir in source_dirs:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png", ".mp4")):
                        media_files.append(os.path.join(root, file))
        return media_files

    @staticmethod
    def __backup_files(media_files: list, backup_directory: str):
        """
        Copies media files to a backup directory with timestamped filenames.
        
        Parameters:
            media_files (list): A list of file paths for media files to be backed up.
            backup_directory (str): Destination directory path for storing backup files.
        
        Behavior:
            - Iterates through each media file in the input list
            - Generates a new filename with current timestamp
            - Attempts to copy each file to the backup directory
            - Logs successful copy operations
            - Logs any errors encountered during file copying
        
        Exceptions:
            Handles and logs any exceptions that occur during file copy process
            Does not interrupt the entire backup process if a single file copy fails
        """
        for src_file in media_files:
            dst_file = os.path.join(
                backup_directory,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_"
                + os.path.basename(src_file),
            )
            try:
                shutil.copy2(str(src_file), str(dst_file))
                log.info(f"Copied {os.path.basename(src_file)} to {dst_file}")
            except Exception as e:
                log.error(f"Failed to copy {src_file}: {str(e)}")

    @classmethod
    @log.function
    def backup(cls):
        """
        Orchestrates the complete media backup process by performing sequential backup operations.
        
        This class method coordinates the backup workflow:
        1. Retrieves default media source directories
        2. Sets a standard backup directory
        3. Ensures the backup directory exists
        4. Collects media files from source directories
        5. Copies media files to the backup directory
        6. Logs the completion of the backup process
        
        Returns:
            None: Performs backup operations without returning a value
        
        Raises:
            OSError: If directory creation or file operations fail
            PermissionError: If insufficient permissions for file/directory operations
        """
        source_dirs = cls.__get_default_paths()
        backup_directory = "MediaBackup"
        cls.__ensure_backup_directory_exists(backup_directory)
        media_files = cls.__collect_media_files(source_dirs)
        cls.__backup_files(media_files, backup_directory)
        log.info("Media backup script completed.")


if __name__ == "__main__":
    Media.backup()
