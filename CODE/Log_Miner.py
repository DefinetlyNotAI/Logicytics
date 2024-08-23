import subprocess
from CODE.Custom_Libraries.Log import Log


def backup_windows_logs():
    try:
        log_type='System'
        backup_file='Logs_backup.csv'
        # Construct the PowerShell command as a single string
        cmd = f'Get-EventLog -LogName "{log_type}" | Export-Csv -Path "{backup_file}" -NoTypeInformation'

        # Use subprocess.Popen to execute the PowerShell command
        process = subprocess.Popen(['powershell.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate(input=cmd)

        if process.returncode != 0:
            raise Exception(f"Failed to backup logs: {stderr.strip()}")

        Log().info(f"Windows logs backed up to {backup_file}")
    except Exception as e:
        Log().error(f"Failed to backup logs: {str(e)}")

    Log().info("Log Miner completed.")
