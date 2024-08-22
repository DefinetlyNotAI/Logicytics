import subprocess

def backup_windows_logs_powershell(log_type='System', backup_file='Logs_backup.csv'):
    try:
        # Construct the PowerShell command as a single string
        cmd = f'Get-EventLog -LogName "{log_type}" | Export-Csv -Path "{backup_file}" -NoTypeInformation'

        # Use subprocess.Popen to execute the PowerShell command
        process = subprocess.Popen(['powershell.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate(input=cmd)

        if process.returncode != 0:
            raise Exception(f"Failed to backup logs: {stderr.strip()}")

        print(f"Windows logs backed up to {backup_file}")
    except Exception as e:
        print(f"Failed to backup logs: {str(e)}")

backup_windows_logs_powershell('System', 'system_logs_backup.csv')

