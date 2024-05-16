import ctypes
import os
import platform
import subprocess
import colorlog

# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level
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

def execute_code(script_path):
    if os.path.splitext(script_path)[1].lower() == '.ps1':
        unblock_command = f'powershell.exe -Command "Unblock-File -Path {script_path}"'
        subprocess.run(unblock_command, shell=True, check=True)
        logger.info("Script unblocked.")

    command = f'powershell.exe -Command "& {script_path}"'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print the output in real-time
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        if decoded_line:
            logger.info(decoded_line)
    
    # Wait for the process to finish and get the final output/error
    stdout, _ = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode('utf-8') if stdout else ""

    # Return the output and error messages
    return stdout, ""

def set_execution_policy():
    command = "powershell.exe Set-ExecutionPolicy Unrestricted -Scope CurrentUser -Force"
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info("Execution policy has been set to Unrestricted.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to set execution policy to Unrestricted. Error: {e}")

def checks():
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if platform.system() == 'Windows':
        if is_admin():
            logger.info("code.py is running with administrative privileges.")
        else:
            logger.warning("code.py is running without administrative privileges. This may cause errors")
    else:
        logger.warning("This script is intended to run on Windows.")

def main():
    set_execution_policy()
    checks()
    for script_path in ["./CMD_Disabled_Bypass.py", "./Copy_System_Files.py", "./Browser_And_Policies_Miner.ps1",
                        "./Window_Features_Lister.bat",
                        "./Antivirus_Finder.ps1", "./IP_Scanner.py", "./Device_Data.bat", "./Sys_Tools.py",
                        "./Tree_Command.bat", "./Simple_Password_Miner.py", "./Copy_Media.py",
                        "./System_Info_Grabber.py", "./Zipper.py"]:
        execute_code(script_path)

if __name__ == "__main__":
    main()
    input("Press Any Button to continue: ")
