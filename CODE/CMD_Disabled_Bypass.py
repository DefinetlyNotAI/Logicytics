import pyautogui
import time
import colorlog
import os
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
    process = subprocess.Popen(r'powershell.exe -Command "& .\Crash_Reporter.py"', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        print(decoded_line)
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()
    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""
    print(stdout)


# Create a logger
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level

# Define a handler that outputs logs to console
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


def press_win_r():
    """
    Simulates pressing the Win+R keys to open the Run dialog
    """
    try:
        pyautogui.hotkey('win', 'r')
        logger.info("Simulated pressing Win+R to open the Run dialog.")
    except Exception as e:
        logger.error(f"Failed to simulate pressing Win+R: {e}")
        crash("IOE", "fun71", e, "error")


def type_command():
    """
    Types the command to enable the command prompt
    """
    try:
        pyautogui.write(
            'cmd.exe /k "REG add HKCU\\Software\\Policies\\Microsoft\\Windows\\System /v DisableCMD /t REG_DWORD /d 0 /f"')
        logger.info("Typed the command to attempt to enable command prompt.")
    except Exception as e:
        logger.error(f"Failed to type the command: {e}")
        crash("IOE", "fun83", e, "error")


def press_enter():
    """
    Presses the Enter key to execute the command
    """
    try:
        pyautogui.press('enter')
        logger.info("Pressed Enter to execute the command.")
    except Exception as e:
        logger.error(f"Failed to press Enter: {e}")
        crash("IOE", "fun96", e, "error")


def press_alt_f4():
    """
    Simulates pressing Alt+F4 to close the command prompt window
    """
    try:
        pyautogui.hotkey('alt', 'f4')
        logger.info("Simulated pressing Alt+F4 to close the command prompt window.")
    except Exception as e:
        logger.error(f"Failed to simulate pressing Alt+F4: {e}")
        crash("IOE", "fun108", e, "error")


if __name__ == "__main__":
    # Wait a bit to ensure the script is ready to run
    time.sleep(2)
    press_win_r()

    # Wait a bit for the Run dialog to appear
    time.sleep(1)
    type_command()
    press_enter()

    # Wait a bit for the command to execute and the command prompt to open
    time.sleep(2)
    press_alt_f4()

    logger.info(
        "Command executed to enable the command prompt and the window has been closed.")
