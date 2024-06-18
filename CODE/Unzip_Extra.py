import os
import zipfile
import colorlog
import subprocess


def crash(error_id, function_no, error_content, type):
    """
    Ensure error_id and function_no are strings
    Prepare the data to write to the temporary files
    Write the name of the placeholder script to the temporary file
    Write the error message to the temporary file
    Write the name of the placeholder function to the temporary file
    Write the name of the placeholder language to the temporary file
    Write the name of the placeholder crash to the temporary file
    Write the type to the temporary file
    Open Crash_Reporter.py in a new shell window
    """
    # Ensure error_id and function_no are strings
    error_id = str(error_id)
    function_no = str(function_no)

    # Prepare the data to write to the temporary files
    script_name = os.path.basename(__file__)
    language = os.path.splitext(__file__)[1][1:]  # Extracting the language part

    # Write the name of the placeholder script to the temporary file
    with open("flag.temp", 'w') as f:
        f.write(script_name)

    # Write the error message to the temporary file
    with open("error.temp", 'w') as f:
        f.write(error_id)

    # Write the name of the placeholder function to the temporary file
    with open("function.temp", 'w') as f:
        f.write(function_no)

    # Write the name of the placeholder language to the temporary file
    with open("language.temp", 'w') as f:
        f.write(language)

    # Write the name of the placeholder crash to the temporary file
    with open("error_data.temp", 'w') as f:
        f.write(error_content)

    with open("type.temp", 'w') as f:
        f.write(type)

    # Open Crash_Reporter.py in a new shell window
    # Note: This command works for Command Prompt.
    # Adjust according to your needs.
    process = subprocess.Popen(r'powershell.exe -Command "& .\Crash_Reporter.py"', shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        print(decoded_line)
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()
    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""
    print(stdout)


# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.DEBUG)  # Set the log level

# Create a StreamHandler for colorlog
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def unzip_extra():
    """
    Function to unzip extra files in the EXTRA directory.
    Automatically unzips all .zip files found in the directory, replacing existing files if required.
    If the EXTRA directory or .zip files are not found, appropriate errors are raised.
    Logs the process and errors using the logger.
    """
    try:
        # Step 1 & 2: Navigate to the parent directory and then to the BACKUP directory
        current_dir = os.getcwd()  # Get the current working directory
        parent_dir = os.path.dirname(current_dir)  # Move up to the parent directory
        extra_dir_path = os.path.join(parent_dir, 'EXTRA')  # Path to the EXTRA directory

        # Check if the EXTRA directory exists
        if not os.path.exists(extra_dir_path):
            raise FileNotFoundError(f"Directory '{extra_dir_path}' not found.")

        # Step 3: List all .zip files in the EXTRA directory and unzip them automatically
        os.chdir(extra_dir_path)  # Change the current working directory to EXTRA
        zip_files = [file for file in os.listdir('.') if file.endswith('.zip')]

        if not zip_files:
            raise FileNotFoundError("No .zip files found in the backup directory.")

        # Step 4: Move back to the parent directory
        os.chdir(parent_dir)

        # Step 5: Unzip the contents into the EXTRA directory, replacing files if required
        if not os.path.exists(extra_dir_path):
            os.makedirs(extra_dir_path)  # Create the EXTRA directory if it doesn't exist

        for zip_file in zip_files:
            with zipfile.ZipFile(os.path.join(extra_dir_path, zip_file), 'r') as zip_ref:
                zip_ref.extractall(extra_dir_path)  # Extracts all the files into EXTRA directory, replacing them if they exist
            logger.info(f"Unzipped {zip_file} into {extra_dir_path}")

            # Delete the ZIP file after successful extraction
            os.remove(os.path.join(extra_dir_path, zip_file))
            logger.info(f"Deleted {zip_file}")

    except Exception as e:
        logger.error(f"An error occurred while unzipping the extra files: {str(e)}")
        crash("OGE", "fun70", str(e), "error")


# Call the function to start unzipping the backups automatically
unzip_extra()
