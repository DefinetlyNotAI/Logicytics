from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from logicytics import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

# List of allowed extensions
allowed_extensions = [
    ".png", ".txt", ".md", ".json", ".yaml", ".secret", ".jpg", ".jpeg",
    ".password", ".text", ".docx", ".doc", ".xls", ".xlsx", ".csv",
    ".xml", ".config", ".log", ".pdf", ".zip", ".rar", ".7z", ".tar"
]


class Mine:
    @staticmethod
    def __search_files_by_keyword(root: Path, keyword: str) -> list:
        """
        Searches for files containing the specified keyword in their names.
        Args:
            root (Path): The root directory to search in.
            keyword (str): The keyword to search for in file names.
        Returns:
            list: List of files that match the search criteria.
        """
        matching_files = []
        for filename in os.listdir(root):
            file_path = root / filename
            if (
                    keyword.lower() in filename.lower()
                    and file_path.is_file()
                    and file_path.suffix in allowed_extensions
            ):
                matching_files.append(file_path)
        return matching_files

    @staticmethod
    def __copy_file(src_file_path: Path, dst_file_path: Path):
        """
        Copies a file to the destination directory.
        Args:
            src_file_path (Path): Source file path.
            dst_file_path (Path): Destination file path.
        Returns:
            None
        """
        try:
            shutil.copy(src_file_path, dst_file_path)
            log.debug(f"Copied {src_file_path} to {dst_file_path}")
        except FileExistsError as e:
            log.warning(f"Failed to copy file due to it already existing: {e}")
        except Exception as e:
            log.error(f"Failed to copy file: {e}")

    def __search_and_copy_files(self, keyword: str):
        """
        Searches for files containing the specified keyword in their names and copies them to a destination directory.
        Args:
            keyword (str): The keyword to search for in file names.
        Returns:
            None
        """
        log.info(f"Searching/Copying file's with keyword: {keyword}")
        drives_root = Path("C:\\")
        destination = Path("Password_Files")
        if not destination.exists():
            destination.mkdir()

        with ThreadPoolExecutor() as executor:
            for root, dirs, files in os.walk(drives_root):
                future_to_file = {executor.submit(self.__search_files_by_keyword, Path(root), keyword): root_path for
                                  root_path in dirs}
                for future in future_to_file:
                    for file_path in future.result():
                        dst_file_path = destination / file_path.name
                        executor.submit(self.__copy_file, file_path, dst_file_path)

    @log.function
    def passwords(self):
        """
        Searches for files containing sensitive data keywords in their filenames,
        copies them to a "Password Files" directory, and logs the completion of the task.
        Returns:
            None
        """
        keywords = ["password", "secret", "code", "login", "api", "key"]

        # Ensure the destination directory is clean
        destination = Path("Password_Files")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()

        for word in keywords:
            self.__search_and_copy_files(word)

        log.info("Sensitive Data Miner Completed")


log.warning("Sensitive Data Miner Started, This may take a while... (aka touch some grass)")
Mine().passwords()
