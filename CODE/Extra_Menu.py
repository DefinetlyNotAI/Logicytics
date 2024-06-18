from CODE.local_libraries.Setups import *


def print_colored(text, color):
    """
    Prints the given text in the specified color.

    :param text: The text to print.
    :param color: The color code (e.g., 'red', 'green', etc.).
    """
    # ANSI escape sequence for resetting the color back to default
    reset = "\033[0m"
    # Mapping of color names to their corresponding ANSI codes
    color_codes = {
        'red': '\033[31m',
    }

    # Check if the color exists and print the colored text
    if color.lower() in color_codes:
        print(color_codes[color.lower()] + text + reset)
    else:
        print("Invalid color name")


def navigate_and_search():
    """
    Navigate to the 'EXTRA' directory within the parent directory and search for specific file types.

    This function searches for .ps1, .exe, or .py files in the 'EXTRA' directory and allows the user to open a selected file.
    """

    # Navigate to the parent directory
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    # Then navigate to the 'EXTRA' directory within the parent directory
    extra_dir_path = os.path.join(parent_dir, 'EXTRA')

    # Check if the 'EXTRA' directory exists
    if not os.path.exists(extra_dir_path):
        print_colored("The 'EXTRA' directory does not exist.", "red")
        exit(1)

    # Search for .ps1, .exe, or .py files
    files_found = []
    for root, dirs, files in os.walk(extra_dir_path):
        for file in files:
            if file.endswith('.ps1') or file.endswith('.exe') or file.endswith('.py'):
                files_found.append(os.path.join(root, file))

    # If no files were found, inform the user
    if not files_found:
        print_colored("No .ps1, .exe, or .py files were found in the specified directory.", "red")
        exit(1)

    # Create a menu for the user to select a file
    print("Select a file to open from the options provided:")
    for index, file_path in enumerate(files_found, start=1):
        print(f"{index}. {os.path.basename(file_path)}")

    choice = input("Enter the number of the file you wish to open: ")

    try:
        choice = int(choice) - 1  # Adjusting for zero-based indexing
        if 0 <= choice < len(files_found):
            selected_file = files_found[choice]
            print()

            # Determine the file type and open accordingly
            _, file_extension = os.path.splitext(selected_file)
            if file_extension.lower() == '.exe':
                subprocess.run([selected_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif file_extension.lower() == '.py':
                subprocess.run(['python', selected_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif file_extension.lower() == '.ps1':
                subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', selected_file],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Exit after processing the file
            print("Exiting after completed processing the file...")
            return True
            # Return to exit the function and thus the script
        else:
            print_colored("Invalid selection. Please choose a valid option.", "red")
            print()
            return False
    except ValueError:
        print_colored("Please enter a valid number.", "red")
        print()
        return False


# Call the function
while navigate_and_search() is not True:
    pass
