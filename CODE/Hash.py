import hashlib
from pathlib import Path
from local_libraries.Setups import *


def hash_zip():
    """
    Calculates the SHA-256 hash of the latest ZIP file in the 'ACCESS/DATA' subdirectory of the parent directory.
    Creates a new file with the same name as the ZIP file but with a '.hash' extension, and writes the hash into it.
    Attempts to make the new file read-only.

    This function does not take any parameters.

    This function does not return any values.

    If no ZIP files are found in the 'ACCESS/DATA' subdirectory, it prints "No ZIP files found."

    If a ZIP file is found, it prints the path of the latest ZIP file found.

    It then calculates the SHA-256 hash of the latest ZIP file and prints the hash.

    It creates a new file with the same name as the ZIP file but with a '.hash' extension, and writes the hash into it.

    It attempts to make the new file read-only. If it fails to do so due to insufficient permissions, it prints "Permission denied to change file permissions."
    """
    # Step 1 & 2: Navigate to the parent directory and access the 'ACCESS/DATA' subdirectory
    parent_dir = Path(
        ".."
    )  # Adjust '..' if necessary to correctly point to the parent directory
    data_dir = parent_dir / "ACCESS/DATA"

    # Step 3: Find the latest ZIP file
    latest_zip = None
    for item in data_dir.glob("**/*.zip"):
        if latest_zip is None or item.stat().st_mtime > latest_zip.stat().st_mtime:
            latest_zip = item

    if latest_zip is None:
        crash("FNF", "fun6", "No ZIP files found.", "crash")
        exit(1)

    # Step 4: Calculate the hash of the ZIP file
    with open(latest_zip, "rb") as f:
        bytes_hash = f.read()
    hash_object = hashlib.sha256(bytes_hash)
    hex_dig = hash_object.hexdigest()

    # Step 5 & 6: Create a new file named after the ZIP file but with a '.hash' extension,
    # write the hash into it, and attempt to make it read-only.
    new_file_name = latest_zip.stem + ".SHA-256"
    new_file_path = data_dir / new_file_name
    with open(new_file_path, "w") as f:
        f.write(hex_dig)

    # Attempting to make the file read-only. Note: This may require administrative privileges.
    try:
        os.chmod(
            new_file_path, 0o444
        )  # 0o444 sets permissions so that the owner has read-only access
    except PermissionError as pe:
        crash("PE", "fun6", pe, "error")


# Call the function
try:
    hash_zip()
except Exception as e:
    print(f"An error occurred: {e}")
    crash("EVE", "fun60", e, "error")
