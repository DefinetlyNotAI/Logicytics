# Python Script Using PyAutoGUI and ColorLog for Automating System Tasks

This Python script demonstrates the automation of several system-level actions using the `pyautogui` library, along with logging capabilities provided by `colorlog`. The script simulates pressing keys and mouse buttons to perform tasks such as opening the Run dialog, typing a command to enable the command prompt, executing the command, and closing the command prompt window. Throughout the process, it uses colored logging to provide feedback on its operations.

## Key Elements

### Importing Libraries

```python
import pyautogui
import time
import colorlog
```

The script begins by importing the necessary libraries:
- `pyautogui`: A module used for programmatically controlling the mouse and keyboard.
- `time`: Provides functions for working with time, including pausing the script execution.
- `colorlog`: Allows for colored terminal output, enhancing readability and distinguishing between different levels of log messages.

### Setting Up Logging

```python
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
```

This section configures a logger with `colorlog` to output log messages in color, making them easier to read and distinguish based on their severity level.

### Functions for Automation

#### Press Win+R

```python
def press_win_r():
    pyautogui.hotkey('win', 'r')
    logger.info("Simulated pressing Win+R to open the Run dialog.")
```

Simulates pressing the Windows key + R to open the Run dialog box.

#### Type Command

```python
def type_command():
    pyautogui.write(
        'cmd.exe /k "REG add HKCU\\Software\\Policies\\Microsoft\\Windows\\System /v DisableCMD /t REG_DWORD /d 0 /f"')
    logger.info("Typed the command to enable the command prompt.")
```

Type a command to enable the command prompt via the Registry Editor.

#### Press Enter

```python
def press_enter():
    pyautogui.press('enter')
    logger.info("Pressed Enter to execute the command.")
```

Simulates pressing the Enter key to execute the previously typed command.

#### Press Alt+F4

```python
def press_alt_f4():
    pyautogui.hotkey('alt', 'f4')
    logger.info("Simulated pressing Alt+F4 to close the command prompt window.")
```

Simulates pressing Alt+F4 to close the currently active window, in this case, the command prompt window opened by the previous commands.

### Main Execution Flow

```python
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
        "INFO: Command executed to enable the command prompt and the window has been closed. Mouse and keyboard have been re-enabled.")
```

The main part of the script waits briefly before starting the sequence of actions. It opens the Run dialog, types a command to enable the command prompt, executes the command, and finally closes the command prompt window. After completing these steps, it logs a final informational message indicating the completion of the task.

## Conclusion

This script showcases how `pyautogui` can be used for automating interactions with the operating system at a low level, combined with `colorlog` for enhanced logging. While this specific example focuses on enabling the command prompt, similar techniques can be applied to automate a wide range of tasks involving keyboard and mouse inputs.
