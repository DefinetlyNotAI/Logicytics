from __lib_actions import *
from __lib_log import *


def backup_windows_logs():
    try:
        log_type = "System"
        backup_file = "Logs_backup.csv"
        # Construct the PowerShell command as a single string
        cmd = f'Get-EventLog -LogName "{log_type}" | Export-Csv -Path "{backup_file}" -NoTypeInformation'

        # Use subprocess.Popen to execute the PowerShell command
        process = subprocess.Popen(
            ["powershell.exe", "-Command", cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate(input=cmd)

        if process.returncode != 0:
            raise Exception(f"Failed to backup logs: {stderr.strip()}")

        log.info(f"Windows logs backed up to {backup_file}")
    except Exception as e:
        log.error(f"Failed to backup logs: {str(e)}")

    log.info("Log Miner completed.")


log = Log(debug=DEBUG)
backup_windows_logs()
