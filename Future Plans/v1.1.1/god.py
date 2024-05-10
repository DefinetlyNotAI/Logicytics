import pyautogui
import subprocess

def open_run_dialog():
    # Press Windows + R
    pyautogui.hotkey('win', 'r')

def type_and_execute_command():
    # Type the command
    command = "shell:::{ED7BA470-8E54-465E-825C-99712043E01C}"
    pyautogui.write(command)
    # Press Enter to execute the command
    pyautogui.press('enter')

def main():
    open_run_dialog()
    type_and_execute_command()

if __name__ == "__main__":
    main()
