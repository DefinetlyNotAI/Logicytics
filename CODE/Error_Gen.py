import re

from local_libraries.Setups import *


def validate_error_id(error_id):
    """
    Validates the given error ID.

    Args:
        error_id (str): The error ID to be validated.

    Raises:
        ValueError: If the error ID is not valid. The error message includes the reason for the invalidity.

    Returns:
        bool: True if the error ID is valid, False otherwise.

    This function checks if the length of the error ID is less than 2 or greater than 4, or if it contains any digits. If any of these conditions are met, a ValueError is raised with an appropriate error message. Otherwise, the function returns True.

    Example:
        validate_error_id("AA") # Returns True
        validate_error_id("A1") # Raises ValueError: "This ID is not valid due to the length being less than 2 or greater than 4 or contains numbers."
    """
    if len(error_id) < 2 or len(error_id) > 4 or re.search(r"\d", error_id):
        raise ValueError(
            "\033[91m"
            + "This ID is not valid due to the length being less than 2 or greater than 4 or contains numbers."
            + "\033[0m"
        )
    return True


def validate_file_name(file_name):
    """
    Validates the given file name and checks if it exists in the file system.

    Args:
        file_name (str): The name of the file to be validated.

    Raises:
        ValueError: If the file name does not include a file extension.

    Returns:
        bool: True if the file name is valid and exists in the file system, False otherwise.

    This function first checks if the file name includes a file extension. If not, it raises a ValueError with a colored error message. Then, it checks if the file exists in the file system. If not, it generates suggestions for similar file names by replacing a single character in the file name with all the lowercase ASCII letters. If no similar file names are found, it prints a colored message indicating that no similar files were found. Otherwise, it prints a colored message with the suggestions. Finally, it returns True if the file name is valid and exists in the file system, False otherwise.
    """
    if "." not in file_name:
        raise ValueError(
            "\033[91m"
            + "Please include the file extension (e.g. hello.html)."
            + "\033[0m"
        )

    # Check if the file exists
    if not os.path.exists(file_name):
        # Generate suggestions for similar filenames
        suggestions = []
        for i in range(len(file_name)):
            for j in range(26):  # Assuming ASCII letters
                new_name = file_name[:i] + chr(ord("a") + j) + file_name[i + 1:]
                if os.path.exists(new_name):
                    suggestions.append(new_name)

        if not suggestions:
            print_colored(f"No similar files found for '{file_name}'.", "yellow")
        else:
            print_colored(
                f"No file named '{file_name}' found. Did you mean one of these?",
                "yellow",
            )
            for suggestion in suggestions:
                print(suggestion)

    return True


def check_file_exists(file_path):
    """
    A function that checks if a file exists at the given file path.

    Parameters:
        file_path (str): The path of the file to be checked.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def read_error_codes(file_path):
    """
    Reads the error codes from a file located at the given file path.

    Parameters:
        file_path (str): The path of the file containing the error codes.

    Returns:
        list: A list of error codes read from the file. Each error code is a string
        with leading and trailing whitespace removed.

    If the file does not exist at the given file path, it will be created and
    an empty file will be returned. The file path is checked using the
    `check_file_exists` function.

    The function opens the file in read mode, reads all lines, and returns a list
    of error codes by stripping leading and trailing whitespace from each line.

    Example usage:
        error_codes = read_error_codes('error_codes.txt')
    """
    if not check_file_exists(file_path):
        print_colored(f"File {file_path} does not exist. Creating a new one.", "red")
        open(file_path, "a").close()
    with open(file_path, "r") as file:
        content = file.readlines()
    return [line.strip() for line in content]


def find_existing_entry(file_path, file_name=None, error_id=None):
    """
    Find an existing entry in a file based on either the file name or the error ID.

    Parameters:
        file_path (str): The path of the file containing the error codes.
        file_name (str, optional): The name of the file to search for. Defaults to None.
        error_id (str, optional): The error ID to search for. Defaults to None.

    Returns:
        str or None: The existing entry found in the file, or None if no match is found.

    This function reads the error codes from the file located at the given file path. It then iterates over each entry in the file and checks if it matches either the file name or the error ID. If a match is found, the function returns the corresponding entry. If no match is found, the function returns None.

    Example usage:
        entry = find_existing_entry('error_codes.txt', file_name='example.txt', error_id='1234')
    """
    existing_entries = read_error_codes(file_path)
    for entry in existing_entries:
        parts = entry.split(" = ")
        if file_name and parts[0] == file_name:
            return entry
        elif error_id and parts[1] == error_id:
            return entry
    return None


def write_new_entry(file_path, file_name, error_id):
    """
    Write a new entry to the specified file.

    Parameters:
        file_path (str): The path of the file to write the new entry to.
        file_name (str): The name of the file.
        error_id (str): The error ID associated with the file.

    Returns:
        None

    This function opens the specified file in 'append' mode and writes a new entry in the format
    "{file_name} = {error_id}\n". The new entry represents a mapping between the file name and
    the corresponding error ID.

    Example usage:
        write_new_entry("error_codes.txt", "example.txt", "1234")
    """
    with open(file_path, "a") as file:
        file.write(f"{file_name} = {error_id}\n")


def main():
    """
    The main function of the program. It performs the following steps:
    1. Retrieve the current working directory and the parent directory.
    2. Constructs the path to the "error.codes" file located in the "SYSTEM" directory.
    3. Call the `list_files_without_error_codes` function to list files without error codes.
    4. Prompts the user to input the file name and error ID.
    5. Validates the input and checks if the file exists.
    6. Checks if the file name or error ID already exists in the "error.codes" file.
    7. Write the new entry to the "error.codes" file.
    8. Repeat steps 4-7 until the user chooses to exit.
    9. Print the total number of successful additions.
    10. Call the `sort_error_codes` function to sort the error codes.

    Parameters:
    None

    Returns:
    None
    """
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    system_dir = os.path.join(parent_dir, "SYSTEM")
    file_path = os.path.join(system_dir, "error.codes")

    # Updated function to list files without error codes, using substring matching
    def list_files_without_error_codes():
        """
        A function that lists files without error codes.
        It retrieves existing entries, all files in the current directory, and calculates files without error codes.
        If there are files without error codes, it prints them with color coding, ensuring they have file extensions.
        Returns False if there are no files without error codes.
        """
        existing_entries = set(
            line.split("=")[0].strip() for line in read_error_codes(file_path)
        )
        all_files = set([f for f in os.listdir(current_dir) if os.path.isfile(f)])
        files_without_error_codes = all_files - existing_entries
        if len(files_without_error_codes) != 0:
            print_colored("Suggested files without error codes:", "green")
            for file in files_without_error_codes:
                _, ext = os.path.splitext(file)
                if ext:  # Ensure there's a file extension
                    print(file)
            return True
        else:
            return False

    print_colored(
        "Hi, welcome to the Error Codes Generator. Let's get started by inputting the file name and error ID you created,",
        "green",
    )
    print_colored(
        "the file name is ANY file you made in the CODE directory while the error ID is a unique ID that you created for that file (Make it Unique).",
        "green",
    )
    print()

    # List files without error codes
    if list_files_without_error_codes():
        successful_additions = 0
        print_colored("\nType exit to quit.", "green")

        while True:
            file_name = input(
                "\nEnter the file name with its extension (e.g., hello.html): "
            )
            error_id = (
                input("Enter a unique error ID (2-4 characters, no numbers): ")
                .strip()
                .upper()
            )

            if file_name == "exit" or error_id == "exit":
                break

            try:
                validate_error_id(error_id)
                validate_file_name(file_name)

                if not os.path.exists(file_name):
                    print_colored(
                        f"The file '{file_name}' doesn't exist in the current working directory.",
                        "red",
                    )
                    continue

                existing_entry_by_name = find_existing_entry(
                    file_path, file_name=file_name
                )
                if existing_entry_by_name:
                    print_colored(
                        f"File name already exists in error.codes: {existing_entry_by_name}",
                        "red",
                    )
                    continue

                existing_entry_by_id = find_existing_entry(file_path, error_id=error_id)
                if existing_entry_by_id:
                    print_colored(
                        f"Error ID already exists in error.codes: {existing_entry_by_id}",
                        "red",
                    )
                    continue

                write_new_entry(file_path, file_name, error_id)
                print(f"Successfully added '{file_name} = {error_id}' to error.codes.")
                successful_additions += 1
            except ValueError as e:
                print(e)

            user_continue = input(
                "\nDo you want to add another file? (yes/no): "
            ).lower()
            if user_continue != "yes":
                break

        print_colored(f"\nTotal successful additions: {successful_additions}", "green")
    else:
        print_colored("No files without error codes found.", "green")

    # Sort error codes after all operations
    sort_error_codes()


def sort_error_codes():
    """
    Sorts the error codes in the 'error.codes' file located in the 'SYSTEM' directory,
    which is one level above the directory containing this script. The function reads the
    lines from the file, sorts them, and then writes them back to the file. If the file is not
    found, a message is printed indicating that the file was not found.

    Parameters:
    None

    Returns:
    None
    """
    base_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(base_path, "..", "SYSTEM", "error.codes")

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        sorted_lines = sorted(lines)

        with open(file_path, "w") as file:
            file.writelines(sorted_lines)

        print()
        print_colored("Error codes have been sorted.", "green")
    except FileNotFoundError:
        print(f"File {file_path} not found.")


if __name__ == "__main__":
    main()
