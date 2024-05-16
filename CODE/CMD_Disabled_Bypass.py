import pyautogui
import time
import colorlog

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


# Function to simulate pressing Win+R to open the Run dialog
def press_win_r():
    pyautogui.hotkey('win', 'r')
    logger.info("Simulated pressing Win+R to open the Run dialog.")


# Function to type the command to enable the command prompt
def type_command():
    pyautogui.write(
        'cmd.exe /k "REG add HKCU\\Software\\Policies\\Microsoft\\Windows\\System /v DisableCMD /t REG_DWORD /d 0 /f"')
    logger.info("Typed the command to enable the command prompt.")


# Function to press Enter to execute the command
def press_enter():
    pyautogui.press('enter')
    logger.info("Pressed Enter to execute the command.")


# Function to simulate pressing Alt+F4 to close the command prompt window
def press_alt_f4():
    pyautogui.hotkey('alt', 'f4')
    logger.info("Simulated pressing Alt+F4 to close the command prompt window.")


# Main execution flow
if __name__ == "__main__":
    # Wait a bit to ensure the script is ready to run
    time.sleep(2)

    press_win_r()

    # Wait a bit for the Run dialog to appear
    time.sleep(1)

    type_command()

    press_enter()

    # Wait a bit for the command to execute and the command prompt to open
    time.sleep(5)

    press_alt_f4()

    # Wait a bit to ensure the command prompt window is closed
    time.sleep(2)

    print("Command executed to enable the command prompt and the window has been closed.")
