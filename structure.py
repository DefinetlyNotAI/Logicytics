from pathlib import Path

# Get the current working directory
current_dir = Path.cwd()

# Define the directories to exclude
exclude_dirs = ['.github', '.idea', '.venv', '_pycache_', 'EXTRA', 'SYSTEM']


# Function to remove excluded directories from the path
def remove_excluded_dirs(path, exclude_dirs):
    for dir in exclude_dirs:
        if path.is_dir() and path.name == dir:
            return path.parent
    return path


# Get the base directory excluding unwanted directories
base_dir = current_dir
for dir in exclude_dirs:
    base_dir = remove_excluded_dirs(base_dir, exclude_dirs)

# Define the fixed beginning part of the path
fixed_beginning = '='

# Get all files in the path, excluding unwanted directories and subdirectories
files = []
for path in Path(base_dir).rglob('*'):
    # Exclude unwanted directories and subdirectories
    if path.is_file():
        # Convert the Path object to a string before concatenating
        relative_path = fixed_beginning + str(path.relative_to(base_dir))
        files.append(relative_path)

# Write the found paths to the output file
output_file = Path(base_dir) / 'Logicystics.structure'
with open(output_file, 'w', encoding='utf-8') as f:
    for file in files:
        f.write(file + '\n')

print(f"Paths have been written to {output_file}.")


# Function to remove lines containing any of the unwanted substrings
def remove_unwanted_lines(file_path, unwanted_substrings):
    with open(file_path, 'r', encoding='utf-8') as read_file:
        lines = read_file.readlines()

    # Filter lines to exclude unwanted substrings
    filtered_lines = [line for line in lines if not any(substring in line for substring in unwanted_substrings)]

    # Write the filtered lines back to the file, replacing its contents
    with open(file_path, 'w', encoding='utf-8') as write_file:
        write_file.writelines(filtered_lines)


# Specify the unwanted substrings
unwanted_substrings = ["EXTRA", "SYSTEM", "_pycache_", ".idea", ".github", ".venv"]

# Remove lines containing any of the unwanted substrings from the output file
remove_unwanted_lines(output_file, unwanted_substrings)

print(
    f"Lines containing any of the following substrings have been removed from {output_file}: {' '.join(unwanted_substrings)}.")
