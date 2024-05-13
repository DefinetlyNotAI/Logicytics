from pathlib import Path

# Get the current working directory
current_dir = Path.cwd()

# Automatically use the parent directory of the current working directory
scan_dir = current_dir.parent

# Get all files in the path, excluding .GitHub,.idea,.venv, EXPLAIN and _pycache directories
files = []
for path in Path(scan_dir).rglob('*'):
    if path.is_file() and not path.name.startswith(('.github', '.idea', '.venv', '_pycache_')):
        files.append(str(path))

# Create a new file with the full paths of the files found
with open(Path(current_dir) / 'Logicystics.structure', 'w', encoding='utf-8') as f:
    for file in files:
        f.write(file + '\n')

# Read the created file and filter out lines containing unwanted strings
with open(Path(current_dir) / 'Logicystics.structure', 'r', encoding='utf-8') as f:
    lines = f.readlines()

filtered_lines = [line for line in lines if not any(unwanted in line for unwanted in ['.venv', '.idea', '.github', '_pycache_', 'SYSTEM'])]

# Write the filtered lines back to the file
with open(Path(current_dir) / 'Logicystics.structure', 'w', encoding='utf-8') as f:
    f.writelines(filtered_lines)

print("The file Logicystics.structure has been updated to exclude lines containing unwanted strings.")
