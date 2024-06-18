import os
import subprocess
import time
import colorlog


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


# Configure colorlog for colored console logs
logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.INFO)  # Set the log level to INFO

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


def run_command_with_timeout(command, timeout=10):
    """
    Executes a command with a specified timeout and returns partial output if the command runs indefinitely.

    Args:
    - command (str): The command to execute.
    - timeout (int): Timeout duration in seconds.

    Returns:
    - tuple: A tuple containing the command output and any error messages.
    """
    try:
        # Start the process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                                   universal_newlines=True)

        # Variables to store output and error messages
        output = ""
        error = ""

        # Wait for the process to complete or timeout
        start_time = time.time()
        while True:
            line = process.stdout.readline()
            if not line:
                break
            # Append each line to the output
            output += line
            # Check for timeout
            if time.time() - start_time > timeout:
                process.terminate()
                break

        # Capture any remaining output and errors
        output += process.stdout.read()
        error += process.stderr.read()

        return output, error
    except Exception as e:
        return f"Error: {str(e)}", ""


def get_netstat_info():
    """
    Retrieves network statistics using the 'netstat' command.
    """
    logger.info("Running netstat -a... Can take around 10 seconds")
    output, error = run_command_with_timeout("netstat -a", timeout=10)
    if error:
        logger.error(f"Error running netstat: {error}")
        crash("EVE", "fun111", error, "error")
    else:
        logger.info("Netstat completed.")
    return output, error


def get_ipconfig_info():
    """
    Retrieves IP configuration using the 'ipconfig' command.
    """
    logger.info("Running ipconfig /all...")
    output, error = run_command_with_timeout("ipconfig /all")
    if error:
        logger.error(f"Error running ipconfig: {error}")
        crash("EVE", "fun125", error, "error")
    else:
        logger.info("Ipconfig completed.")
    return output, error


def get_wifi_profiles():
    """
    Retrieves Wi-Fi profiles using the 'netsh wlan show profiles' command.
    """
    logger.info("Running netsh wlan show profiles...")
    output, error = run_command_with_timeout("netsh wlan show profiles", timeout=15)
    if error:
        logger.error(f"Error running netsh: {error}")
        crash("EVE", "fun139", error, "error")
    else:
        logger.info("Netsh completed.")
    return output, error


def main():
    """
    Main function to orchestrate the retrieval and saving of network information.
    """
    # Run each command and gather the outputs
    netstat_output, netstat_error = get_netstat_info()
    ipconfig_output, ipconfig_error = get_ipconfig_info()
    wifi_profiles_output, wifi_profiles_error = get_wifi_profiles()

    # Save the outputs to a file with clear separation
    filename = "network_info.txt"
    with open(filename, 'w') as file:
        file.write("=== Network Statistics ===\n")
        file.write(netstat_output + "\n\n")
        file.write("=== IP Configuration ===\n")
        file.write(ipconfig_output + "\n\n")
        file.write("=== WiFi Profiles ===\n")
        file.write(wifi_profiles_output + "\n\n")

    logger.info(f"All information has been saved to {filename}.")


if __name__ == "__main__":
    main()
