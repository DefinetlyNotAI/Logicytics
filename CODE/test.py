import sys

from Libs.__lib_actions import *
from Libs.__lib_log import Log


def set_execution_policy(Silent: str) -> None:
    """
    Sets the PowerShell execution policy to Unrestricted for the current process.

    Args:
        Silent (str): If "Silent", suppresses logging of success or failure.

    Raises:
        subprocess.CalledProcessError: If there was an error setting the execution policy.

    Returns:
        None
    """
    # Define the command to set the execution policy
    command = 'powershell.exe -Command "Set-ExecutionPolicy Unrestricted -Scope Process -Force"'

    if Silent == "Debug":
        log.debug("Command: " + command)

    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if Silent == "Debug":
            log.debug(f"Result: {result}")

        # Check the output for success
        if "SUCCESS" in result.stdout:
            if Silent != "Silent":
                log.info("Execution policy has been set to Unrestricted.")
        else:
            log.error("An error occurred while trying to set the execution policy.")
            print("OSE", "fun274", "Not able to set execution policy", "error")

    except subprocess.CalledProcessError as e:
        log.error(f"An error occurred while trying to set the execution policy: {e}")
        print("EVE", "fun274", e, "error")


log = Log()

import subprocess


def execute_and_log(script):
    result = subprocess.Popen(['python', script], stdout=subprocess.PIPE).communicate()[0]
    if script.endswith('.py'):
        print(result.decode())
    else:
        lines = result.decode().splitlines()
        ID = next((line.split(':')[0].strip() for line in lines if ':' in line), None)
        getattr(log.getLogger(), ID.upper())(result.decode())

# Usage example
execute_and_log('test2.py')
