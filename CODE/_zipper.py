import hashlib
import os
import shutil
import zipfile
from datetime import date


def get_files_to_zip(path: str) -> list:
    return [
        f for f in os.listdir(path)
        if not f.endswith((".py", ".exe", ".bat", ".ps1"))
           and not f.startswith(("config.", "SysInternal_Suite", "__pycache__"))
    ]


def create_zip_file(path: str, files: list, filename: str):
    with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
        for file in files:
            zip_file.write(os.path.join(path, file))


def remove_files(path: str, files: list):
    for file in files:
        try:
            shutil.rmtree(os.path.join(path, file))
        except OSError:
            os.remove(os.path.join(path, file))
        except Exception as e:
            print(e)


def generate_sha256_hash(filename: str) -> str:
    with open(f"{filename}.zip", "rb") as zip_file:
        zip_data = zip_file.read()
    return hashlib.sha256(zip_data).hexdigest()


def write_hash_to_file(filename: str, sha256_hash: str):
    with open(f"{filename}.hash", "w") as hash_file:
        hash_file.write(sha256_hash)


def move_files(filename: str):
    shutil.move(f"{filename}.zip", "../ACCESS/DATA/Zip")
    shutil.move(f"{filename}.hash", "../ACCESS/DATA/Hashes")


def zip_and_hash(path: str, name: str, action: str) -> tuple:
    today = date.today()
    filename = f"Logicytics_{name}_{action}_{today.strftime('%Y-%m-%d')}"
    files_to_zip = get_files_to_zip(path)
    create_zip_file(path, files_to_zip, filename)
    remove_files(path, files_to_zip)
    sha256_hash = generate_sha256_hash(filename)
    write_hash_to_file(filename, sha256_hash)
    move_files(filename)
    return (
        f"Zip file moved to ../ACCESS/DATA/Zip/{filename}.zip",
        f"SHA256 Hash file moved to ../ACCESS/DATA/Hashes/{filename}.hash",
    )
