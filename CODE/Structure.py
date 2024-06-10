# Don't use this script as it is used to update the Logicytics.structure file in the SYSTEM directory,
# based on the files in the CODE directory.

import os
from pathlib import Path  # Use pathlib for more robust path manipulation


def update_structure_file():
    """
    Update the Logicytics.structure file in the SYSTEM directory based on the files in the CODE directory.
    Include paths of all subdirectories and their contents within the CODE directory, ensuring only those with extensions are included.

    The Logicytics.structure file contains the relative paths of all files and directories in the CODE directory,
    with the parent directory part replaced by '='.
    The paths are written to the file with each path on a new line.

    The script assumes that it is located directly inside the CODE directory.
    """

    # Determine the absolute path of the current script
    current_script_directory = Path(__file__).resolve().parent.parent
    script_parent_directory = current_script_directory

    # Construct the directories based on the common parent directory
    code_directory = script_parent_directory / 'CODE'
    system_directory = script_parent_directory / 'SYSTEM'

    # Ensure the system directory exists; create it if it doesn't
    system_directory.mkdir(parents=True, exist_ok=True)

    # Function to traverse directories and collect paths
    def collect_paths(directory):
        paths = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                full_path = Path(root) / name
                rel_path = full_path.relative_to(script_parent_directory)
                if '.' in str(rel_path):  # Check if the path includes a file extension
                    paths.append('=' + str(rel_path).replace(str(script_parent_directory.name) + '/', ''))
            for name in dirs:
                full_path = Path(root) / name
                rel_path = full_path.relative_to(script_parent_directory)
                if '.' in str(rel_path):  # Check if the path includes a file extension
                    paths.append('=' + str(rel_path).replace(str(script_parent_directory.name) + '/', ''))
        return paths

    # Collect paths of all files and directories in the CODE directory
    all_paths = collect_paths(code_directory)

    # Create or overwrite the Logicytics.structure file in the SYSTEM directory
    with open(system_directory / 'Logicytics.structure', 'w') as file:
        for path in all_paths:
            file.write(path + '\n')

    print("The Logicytics.structure file has been updated successfully.")


if __name__ == "__main__":
    update_structure_file()
