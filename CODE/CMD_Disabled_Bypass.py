import pyautogui
import time
from local_libraries.Setups import *


def press_win_r():
    """
    Simulates pressing the Win+R keys to open the Run dialog
    """
    try:
        pyautogui.hotkey('win', 'r')
        logger.info("Simulated pressing Win+R to open the Run dialog.")
    except Exception as e:
        logger.error(f"Failed to simulate pressing Win+R: {e}")
        crash("IOE", "fun6", e, "error")


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
        crash("IOE", "fun18", e, "error")


def press_enter():
    """
    Presses the Enter key to execute the command
    """
    try:
        pyautogui.press('enter')
        logger.info("Pressed Enter to execute the command.")
    except Exception as e:
        logger.error(f"Failed to press Enter: {e}")
        crash("IOE", "fun31", e, "error")


def press_alt_f4():
    """
    Simulates pressing Alt+F4 to close the command prompt window
    """
    try:
        pyautogui.hotkey('alt', 'f4')
        logger.info("Simulated pressing Alt+F4 to close the command prompt window.")
    except Exception as e:
        logger.error(f"Failed to simulate pressing Alt+F4: {e}")
        crash("IOE", "fun43", e, "error")


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
