import os
import shutil
from datetime import date
import zipfile
import hashlib
import sys



def zip_and_hash(path: str, name: str, action: str) -> tuple:
    """
    Zips the files in the given path, excluding those with specific extensions or starting with specific prefixes.
    Generates an SHA-256 hash of the resulting zip file and writes it next to the zip file.
    Moves the zip file and its hash to the ../ACCESS/DATA/Zip and ../ACCESS/DATA/Hashes directories, respectively.

    Parameters:
        path (str): The path of the directory to be zipped.
        name (str): The name of the zip file.
        action (str): The action performed on the files (e.g., "backup", "archive").

    Returns:
        tuple: A tuple containing the paths of the moved zip file and its hash file.
    """
    today = date.today()
    filename = f"Logicytics_{name}_{action}_{today.strftime('%Y-%m-%d')}"

    # Zip files
    files_to_zip = [
        f
        for f in os.listdir(path)
        if not f.endswith((".py", ".exe", ".bat", ".ps1"))
        and not f.startswith(("config.", "SysInternal_Suite", "__pycache__"))
    ]

    with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
        for file in files_to_zip:
            zip_file.write(os.path.join(path, file))

    for file in files_to_zip:
        try:
            shutil.rmtree(os.path.join(path, file))
        except OSError:
            os.remove(os.path.join(path, file))
        except Exception as e:
            print(e)
    # Generate SHA-256 hash of the zip file
    with open(f"{filename}.zip", "rb") as zip_file:
        zip_data = zip_file.read()
    sha256_hash = hashlib.sha256(zip_data).hexdigest()

    # Write hash next to the zip file
    with open(f"{filename}.hash", "w") as hash_file:
        hash_file.write(sha256_hash)

    # Move files
    shutil.move(f"{filename}.zip", "../ACCESS/DATA/Zip")
    shutil.move(f"{filename}.hash", "../ACCESS/DATA/Hashes")

    return (
        f"Zip file moved to ../ACCESS/DATA/Zip/{filename}.zip",
        f"SHA256 Hash file moved to ../ACCESS/DATA/Hashes/{filename}.hash",
    )
