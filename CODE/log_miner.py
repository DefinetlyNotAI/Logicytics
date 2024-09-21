from __lib_class import *

log = Log(debug=DEBUG)


def backup_windows_logs():
    """
    Backs up Windows system logs to a CSV file.

    This function constructs a PowerShell command to retrieve system logs and export them to a CSV file.
    It then executes the command using subprocess.Popen and handles any errors that may occur.
    The function logs the result of the backup operation and any errors that occur.

    Returns:
        None
    """
    try:
        log_type = "System"
        backup_file = "Logs_backup.csv"
        # Construct the PowerShell command as a single string
        cmd = f'Get-EventLog -LogName "{log_type}" | Export-Csv -Path "{backup_file}" -NoTypeInformation'

        # Use subprocess.Popen to Execute the PowerShell command
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


backup_windows_logs()
