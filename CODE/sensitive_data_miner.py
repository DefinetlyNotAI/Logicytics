import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from logicytics import log

# List of allowed extensions
allowed_extensions = [
    ".png", ".txt", ".md", ".json", ".yaml", ".secret", ".jpg", ".jpeg",
    ".password", ".text", ".docx", ".doc", ".xls", ".xlsx", ".csv",
    ".xml", ".config", ".log", ".pdf", ".zip", ".rar", ".7z", ".tar",
    ".gz", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst",
    ".sql", ".db", ".dbf", ".sqlite", ".sqlite3", ".bak", ".dbx",
    ".mdb", ".accdb", ".pst", ".ost", ".msg", ".eml", ".vsd",
    ".vsdx", ".vsdm", ".vss", ".vssx", ".vssm", ".vst", ".vstx",
    ".vstm", ".vdx", ".vsx", ".vtx", ".vdw", ".vsw", ".vst",
    ".mpp", ".mppx", ".mpt", ".mpd", ".mpx", ".mpd", ".mdf",
]


class Mine:
    @staticmethod
    def __search_files_by_keyword(root: Path, keyword: str) -> list:
        """
        Searches for files containing a specified keyword in their names within a given directory.
        
        Parameters:
            root (Path): The root directory to search in for files.
            keyword (str): The keyword to search for in file names (case-insensitive).
        
        Returns:
            list: A list of file paths matching the search criteria, which:
                - Contain the keyword in their filename (case-insensitive)
                - Are files (not directories)
                - Have file extensions in the allowed_extensions list
        
        Raises:
            WindowsError: If permission is denied when accessing the directory (logged as a warning in debug mode)
        
        Notes:
            - Skips files with unsupported extensions, logging debug information
            - Uses case-insensitive keyword matching
        """
        matching_files = []
        path_list = []
        try:
            path_list = os.listdir(root)
        except (WindowsError, PermissionError) as e:
            log.warning(f"Permission Denied: {e}")
        except Exception as e:
            log.error(f"Failed to access directory: {e}")

        for filename in path_list:
            file_path = root / filename
            if (
                    keyword.lower() in filename.lower()
                    and file_path.is_file()
                    and file_path.suffix in allowed_extensions
            ):
                matching_files.append(file_path)
            else:
                log.debug(f"Skipped {file_path}, Unsupported due to {file_path.suffix} extension")
        return matching_files

    @staticmethod
    def __copy_file(src_file_path: Path, dst_file_path: Path):
        """
        Copy a file from the source path to the destination path.
        
        Parameters:
            src_file_path (Path): The full path of the source file to be copied.
            dst_file_path (Path): The full path where the file will be copied.
        
        Raises:
            FileExistsError: If a file already exists at the destination path.
            Exception: For any other unexpected errors during file copying.
        
        Notes:
            - Uses shutil.copy() for file copying
            - Logs debug message on successful copy
            - Logs warning if file already exists
            - Logs error for any other copying failures
        """
        try:
            # Check file size and permissions
            if src_file_path.stat().st_size > 10_000_000:  # 10MB limit
                log.warning("File exceeds size limit")
                return
            shutil.copy(src_file_path, dst_file_path)
            log.debug(f"Copied {src_file_path} to {dst_file_path}")
        except FileExistsError as e:
            log.warning(f"File already exists in destination: {e}")
        except Exception as e:
            log.error(f"Failed to copy file: {e}")

    @classmethod
    def __search_and_copy_files(cls, keyword: str):
        """
        Searches for files containing the specified keyword in their names and copies them to a destination directory.
        
        This method performs a comprehensive file search across the C: drive, identifying files that match a given keyword and concurrently copying them to a dedicated "Password_Files" directory.
        
        Parameters:
            keyword (str): The keyword to search for in file names. Used to filter and identify potentially sensitive files.
        
        Side Effects:
            - Creates a "Password_Files" directory if it does not exist
            - Logs informational messages about the search and copy process
            - Utilizes multithreading to efficiently search and copy files
        
        Notes:
            - Searches recursively through all directories starting from C:\
            - Uses ThreadPoolExecutor for concurrent file searching and copying
            - Handles potential permission and file access errors during search
        """
        log.info(f"Searching/Copying file's with keyword: {keyword}")
        drives_root = Path("C:\\")
        destination = Path("Password_Files")
        if not destination.exists():
            destination.mkdir()

        with ThreadPoolExecutor() as executor:
            for root, dirs, _ in os.walk(drives_root):
                future_to_file = {
                    executor.submit(cls.__search_files_by_keyword, Path(root) / sub_dir, keyword): sub_dir
                    for sub_dir in dirs
                }
                for future in future_to_file:
                    for file_path in future.result():
                        dst_file_path = destination / file_path.name
                        executor.submit(cls.__copy_file, file_path, dst_file_path)

    @classmethod
    @log.function
    def passwords(cls):
        """
        Searches for and copies files containing sensitive data keywords to a dedicated directory.
        
        This method performs a comprehensive search for files with predefined sensitive keywords in their names, 
        copying matching files to a "Password_Files" directory. It handles directory cleanup and uses predefined 
        keywords related to sensitive information.

        Side Effects:
            - Creates or recreates the "Password_Files" directory
            - Copies files matching sensitive keywords to the destination directory
            - Logs the completion of the sensitive data mining process
        
        Keywords Searched:
            - "password"
            - "secret"
            - "code"
            - "login"
            - "api"
            - "key"
        
        Logging:
            - Logs an informational message upon completion of the search and copy process
        """
        keywords = ["password", "secret", "code", "login", "api", "key",
                    "token", "auth", "credentials", "private", "cert", "ssh", "pgp", "wallet"]

        # Ensure the destination directory is clean
        destination = Path("Password_Files")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()

        for word in keywords:
            cls.__search_and_copy_files(word)

        log.info("Sensitive Data Miner Completed")


if __name__ == "__main__":
    log.warning(
        "Sensitive Data Miner Initialized. Processing may take a while... (Consider a break: coffee or fresh air recommended!)")
    Mine.passwords()
