import os
import tkinter as tk
from tkinter import messagebox
from CODE.local_libraries.Lists_and_variables import *
from datetime import datetime

time = datetime.now().strftime('%Y-%m-%d_at_time_%H-%M-%S')


def gui_crash_msg(err_title, err_msg):
    """
    Displays an error message in a GUI window.

    Parameters:
    - err_title: The title of the popup window.
    - err_msg: The content of the popup window.
    """
    # Create the main window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Remove leading/trailing whitespace from err_msg
    err_msg = err_msg.strip()

    # Display the error message using messagebox
    messagebox.showerror(title=f"⚠️ {err_title}", message=err_msg)


def extract_code_from_error_codes(script_name):
    """
    Extracts the code part from the error codes file based on the given script name.

    Args:
        script_name (str): The name of the script.

    Returns:
        str or None: The code part extracted from the error codes file, or None if the script name is not found.

    Raises:
        None.

    Notes:
        - This function assumes that the error codes file is located in the parent directory of the current directory.
        - The error codes file is expected to be in the format of `script_name = code_part`.
        - If the error codes file is not found, no error message is printed.
        - If any other error occurs, an error message is printed.

    """
    parent_dir = '..'
    error_codes_path = os.path.join(parent_dir, 'SYSTEM', 'error.codes')

    try:
        with open(error_codes_path, 'r') as file:
            content = file.read()
            filename_only = os.path.splitext(script_name)[0]

            lines = content.split('\n')
            line_with_filename = next((line for line in lines if filename_only in line), None)

            if line_with_filename is None:
                return None

            code_part = line_with_filename.split('=')[1].strip()
            return code_part

    except FileNotFoundError:
        pass  # No need to print the error message here
    except Exception as e:
        print("An error occurred:", e)
        pass  # Suppressing the exception message


def save_contents_to_list():
    """
    Saves the contents of temporary files into a list and returns it.

    This function reads the contents of temporary files specified in the `temp_files` list.
    It opens each file, reads its contents, and appends it to the `contents_list`.
    After reading the contents, the temporary files are deleted using `os.remove`.
    The function returns the `contents_list` containing the contents of all the temporary files.

    Returns:
        list: A list containing the contents of the temporary files.

    """
    temp_files = ['flag.temp', 'error.temp', 'function.temp', 'language.temp', 'error_data.temp']
    contents_list = []

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                contents_list.append(f.read().strip())
            os.remove(temp_file)  # Delete the temporary file

    return contents_list


def temporary_file_scanner():
    """
    Scans for a temporary file named 'flag.temp' and retrieves its contents.

    This function checks if a temporary file named 'flag.temp' exists in the current directory.
    If the file exists, it opens the file, reads its contents, and removes the file.
    The contents of the file, which is expected to be a script name, are returned.
    If the file does not exist, the function returns None.

    Returns:
        str or None: The contents of 'flag.temp' if it exists, otherwise None.
    """
    temp_file_path = "flag.temp"
    if os.path.exists(temp_file_path):
        with open(temp_file_path, 'r') as f:
            script_name = f.read().strip()
        os.remove(temp_file_path)
        return script_name
    else:
        return None  # Return None if no flag file found


def read_type():
    """
    Reads the contents of a temporary file named 'type.temp' and deletes it after reading.

    This function attempts to open and read the contents of a file named 'type.temp' in the current directory.
    If the file exists, it reads its contents and returns them. If the file does not exist, it prints an error message
    and returns None.

    Returns:
        str or None: The contents of the 'type.temp' file if it exists, otherwise None.
    """
    # Step 1 & 2: Open the file and read its contents
    try:
        with open('type.temp', 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("The file 'error.temp' does not exist.")
        return

    # Step 3: The file is automatically closed after exiting the 'with' block
    # Step 3: Delete the file
    try:
        os.remove('type.temp')
    except OSError as e:
        print(f"Error deleting file: {e.strerror}")

    if content != "":
        return content
    else:
        return "False"


def write_logs(err_title, err_msg):
    logs_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACCESS", "LOGS",
                                  file_err_title)

    with open(logs_file_path, 'w') as file:
        file.write(err_title)
        file.write(err_msg)


if __name__ == "__main__":
    file_name = temporary_file_scanner()
    file_code = extract_code_from_error_codes(file_name)

    contents_list = save_contents_to_list()

    result = {
        'file_code': file_code,
        'temp_contents': contents_list
    }

    # Extracting the 'file_code' value directly since it's not inside a list
    file_code = result['file_code']

    # The 'temp_contents' key contains a list of strings, so we join them with '-'
    temp_contents_str = '-'.join(result['temp_contents'])

    # Constructing the final string
    full_code = f"{file_code}-{temp_contents_str}"

    # Splitting the string by the hyphen ("-")
    split_output = full_code.split("-")

    # Accessing the first "text" (the second element after splitting)
    dic_code = split_output[1]
    err_line = split_output[2]
    file_type = split_output[3]
    err_data = split_output[4]

    # Path to the error.dictionary file
    file_path = os.path.join(os.pardir, 'SYSTEM', 'error.dictionary')

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into words
            words = line.split()
            # Check if the dic_code matches the first word in the line
            if words[0] == dic_code:
                # Print the entire line
                dic_code_full = line.strip()

    try:
        if dic_code_full == "":
            dic_code_full = "UKE = Unknown Error = an unknown error occurs, or something unexpected occurs. This is a special case, where the reporter also received the error!"
    except Exception as e:
        dic_code_full = "UKE = Unknown Error = an unknown error occurs, or something unexpected occurs. This is a special case, where the reporter also received the error!"

    # Now, splitting the string into a list based on the "=" delimiter should work without issues
    split_list = dic_code_full.split('=')

    # Since the string is guaranteed to have at least one "=", we can safely assign the parts to variables
    first_part, errtype, message = split_list

    # Set the results
    process = file_name
    err_line_og = err_line
    err_line_number = err_line.removeprefix("fun")

    try:
        language = languages[file_type]
    except Exception as e:
        language = "undetermined"

    crash_data = read_type()
    if crash_data == "crash":
        err_title = f"Fatal{errtype}({first_part}) at {time}"
        file_err_title = f"{time}_crash_log"
        err_msg = f'''
        
        A Crash Occurred!! 
        At the time {time}, with the file {file_name}, it crashed with the error "{errtype}" which typically occurs when{message}
        We deeply apologize for this, it is suggested to report these errors to GitHub issues in our repository
        if you deem that this error is the codes fault, More details of this error are given below:
        
        The full error code is {full_code}
        The errors type is "{errtype}" which occurred in the line {err_line_number} with the line code being {err_line_og}
        The code is written in the {language} language,
        
        This Crash will also have some extra relevant info from the {file_name} itself, 
        it is highly recommended to create an issue with this file in the ACCESS/LOGS directory, it will be named {file_err_title},

        We were able to get the following raw data,

        {err_data}
        '''
        write_logs(err_title, err_msg)
        gui_crash_msg(err_title, err_msg)

    else:
        err_title = f"Error{errtype}({first_part}) at {time}"
        file_err_title = f"{time}_error_log"
        err_msg = f'''
        
        An error occurred at the time {time}, with the file {file_name}, it crashed with the error "{errtype}" which typically occurs when{message}
        We deeply apologize for this, it is suggested to report these errors to GitHub issues in our repository
        if you deem that this error is the codes fault, More details of this error are given below:

        The full error code is {full_code}
        The errors type is "{errtype}" which occurred in the line {err_line_number} with the line code being {err_line_og}
        The code is written in the {language} language,
        
        It is highly recommended to create an issue with this file in the ACCESS/LOGS directory, it will be named {file_err_title},
        
        We were able to get the following raw data,
        
        {err_data}
        '''
        write_logs(err_title, err_msg)
