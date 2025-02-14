import subprocess

from logicytics import log


@log.function
def backup_windows_logs():
    """
    Backs up Windows system logs to a CSV file using PowerShell.
    
    This function retrieves system logs and exports them to a CSV file named 'Logs_backup.csv'. 
    It uses PowerShell's Get-EventLog cmdlet to collect system logs and Export-Csv to save them.
    
    The function handles potential errors during log backup and logs the operation's outcome.
    If the backup fails, an error message is logged without raising an exception.
    
    Returns:
        None
    
    Raises:
        No explicit exceptions are raised; errors are logged instead.
    
    Example:
        When called, this function will create a 'Logs_backup.csv' file 
        containing all system event log entries.
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
            log.error(f"Failed to backup logs: {stderr.strip()}")

        log.info(f"Windows logs backed up to {backup_file}")
    except Exception as e:
        log.error(f"Failed to backup logs: {str(e)}")

    log.info("Log Miner completed.")


if __name__ == "__main__":
    backup_windows_logs()
