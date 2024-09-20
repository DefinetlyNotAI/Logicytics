import hashlib
import shutil
from datetime import datetime
from __lib_class import *

log_zipper = Log(debug=DEBUG)
log_zipper_funcs = {
    "INFO": log_zipper.info,
    "WARNING": log_zipper.warning,
    "ERROR": log_zipper.error,
    "CRITICAL": log_zipper.critical,
    None: log_zipper.debug,
}


class Zip:
    @staticmethod
    def __get_files_to_zip(path: str) -> list:
        return [
            f
            for f in os.listdir(path)
            if not f.endswith((".py", ".exe", ".bat", ".ps1")) and not f.startswith(
                ("config.", "SysInternal_Suite", "__pycache__"))
        ]

    @staticmethod
    def __create_zip_file(path: str, files: list, filename: str):
        with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
            for file in files:
                zip_file.write(os.path.join(path, file))

    @staticmethod
    def __remove_files(path: str, files: list):
        for file in files:
            try:
                shutil.rmtree(os.path.join(path, file))
            except OSError:
                os.remove(os.path.join(path, file))
            except Exception as e:
                log_zipper.error(e)

    @staticmethod
    def __generate_sha256_hash(filename: str) -> str:
        with open(f"{filename}.zip", "rb") as zip_file:
            zip_data = zip_file.read()
        return hashlib.sha256(zip_data).hexdigest()

    @staticmethod
    def __write_hash_to_file(filename: str, sha256_hash: str):
        with open(f"{filename}.hash", "w") as hash_file:
            hash_file.write(sha256_hash)

    @staticmethod
    def __move_files(filename: str):
        shutil.move(f"{filename}.zip", "../ACCESS/DATA/Zip")
        shutil.move(f"{filename}.hash", "../ACCESS/DATA/Hashes")

    def and_hash(self, path: str, name: str, flag: str) -> tuple:
        today = datetime.now()
        filename = f"Logicytics_{name}_{flag}_{today.strftime('%Y-%m-%d_%H-%M-%S')}"
        files_to_zip = self.__get_files_to_zip(path)
        self.__create_zip_file(path, files_to_zip, filename)
        self.__remove_files(path, files_to_zip)
        sha256_hash = self.__generate_sha256_hash(filename)
        self.__write_hash_to_file(filename, sha256_hash)
        self.__move_files(filename)
        return (
            f"Zip file moved to ../ACCESS/DATA/Zip/{filename}.zip",
            f"SHA256 Hash file moved to ../ACCESS/DATA/Hashes/{filename}.hash",
        )
