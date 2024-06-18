import os
from datetime import datetime
import re


def get_file_age(file_name):
    """
    Extracts the date from the file name and calculates the age of the file.
    """
    match = re.search(r'\d{4}-\d{2}-\d{2}', file_name)
    if match:
        file_date = datetime.strptime(match.group(), '%Y-%m-%d')
        return file_date
    return None


def should_delete_file(file_name, file_size, total_files):
    """
    Determines whether a file should be deleted based on its size, the total number of files,
    and specific criteria such as including 'crash', 'error', or having a '.md' extension.
    Returns a tuple with a boolean and a reason string.
    """
    file_age = get_file_age(file_name)

    if file_age is None:
        return False, "No date found in file name."

    # Calculate the difference between the current date and the file's creation date
    today = datetime.now()
    file_days_old = (today - file_age).days

    # Check for file size limit
    if file_size > 5 * 1024 * 1024:  # Larger than 5MB
        return True, "File size larger than 5MB."

    # Check for total number of files
    if total_files > 30:
        return True, "Total number of files exceeds 30."

    # Apply specific deletion criteria
    if 'crash' in file_name.lower():
        if file_days_old > 90:
            return True, "Contains 'crash' and older than 90 days."
    elif 'error' in file_name.lower():
        if file_days_old > 30:
            return True, "Contains 'error' and older than 30 days."
    elif file_name.endswith('.md'):
        if file_days_old > 20:
            return True, "Has '.md' extension and older than 20 days."

    return False, "No deletion criteria met."


def manage_logs_directory(logs_dir_path):
    """
    Manages the LOGS directory according to the specified rules.
    """
    file_sizes = []
    total_files = 0
    files_to_delete = []

    for root, dirs, files in os.walk(logs_dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_sizes.append(file_size)

            total_files += 1

            should_delete, reason = should_delete_file(file, file_size, total_files)
            if should_delete:
                files_to_delete.append((file_path, reason))

    # Delete files based on the determined criteria
    for file_path, reason in sorted(files_to_delete):
        # print(f"Deleting {file_path} because: {reason}")
        os.remove(file_path)


# Example usage
script_dir = os.path.dirname(os.path.realpath(__file__))
logs_dir_path = os.path.join(script_dir, '..', 'ACCESS', 'LOGS')
manage_logs_directory(logs_dir_path)
