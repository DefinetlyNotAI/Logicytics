import pyautogui
import time

# Wait a bit to ensure the script is ready to run
time.sleep(2)

# Simulate pressing Win+R to open the Run dialog
pyautogui.hotkey('win', 'r')

# Wait a bit for the Run dialog to appear
time.sleep(1)

# Type the command to enable the command prompt
pyautogui.write('cmd.exe /k "REG add HKCU\Software\Policies\Microsoft\Windows\System /v DisableCMD /t REG_DWORD /d 0 /f"')

# Press Enter to execute the command
pyautogui.press('enter')

# Wait a bit for the command to execute and the command prompt to open
time.sleep(5)

# Simulate pressing Alt+F4 to close the command prompt window
pyautogui.hotkey('alt', 'f4')

# Wait a bit to ensure the command prompt window is closed
time.sleep(2)

print("Command executed to enable the command prompt and the window has been closed.")
