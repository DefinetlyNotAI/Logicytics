import os
import zipfile


def unzip(zip_path):
    # Get the base name of the zip file
    base_name = os.path.splitext(os.path.basename(zip_path))[0]

    # Create a new directory with the same name as the zip file
    output_dir = os.path.join(os.path.dirname(zip_path), base_name)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(path=str(output_dir))

