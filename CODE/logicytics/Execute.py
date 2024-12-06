from __future__ import annotations

import subprocess
from subprocess import CompletedProcess


class Execute:
    @classmethod
    def script(cls, script_path: str) -> list[list[str]] | None:
        """
        Executes a script file and handles its output based on the file extension.
        Parameters:
            script_path (str): The path of the script file to be executed.
        """
        if script_path.endswith(".py"):
            cls.__run_python_script(script_path)
            return None
        else:
            if script_path.endswith(".ps1"):
                cls.__unblock_ps1_script(script_path)
            return cls.__run_other_script(script_path)

    @staticmethod
    def command(command: str) -> str:
        """
        Runs a command in a subprocess and returns the output as a string.

        Parameters:
            command (str): The command to be executed.

        Returns:
            CompletedProcess.stdout: The output of the command as a string.
        """
        process = subprocess.run(command, capture_output=True, text=True)
        return process.stdout

    @staticmethod
    def __unblock_ps1_script(script: str):
        """
        Unblocks and runs a PowerShell (.ps1) script.
        Parameters:
            script (str): The path of the PowerShell script.
        Returns:
            None
        """
        try:
            unblock_command = f'powershell.exe -Command "Unblock-File -Path {script}"'
            subprocess.run(unblock_command, shell=False, check=True)
        except Exception as err:
            exit(f"Failed to unblock script: {err}")

    @staticmethod
    def __run_python_script(script: str):
        """
        Runs a Python (.py) script.
        Parameters:
            script (str): The path of the Python script.
        Returns:
            None
        """
        result = subprocess.Popen(
            ["python", script], stdout=subprocess.PIPE
        ).communicate()[0]
        # LEAVE AS PRINT
        print(result.decode())

    @classmethod
    def __run_other_script(cls, script: str) -> list[list[str]]:
        """
        Runs a script with other extensions and logs output based on its content.
        Parameters:
            script (str): The path of the script.
        Returns:
            None
        """
        result = cls.command(f"powershell.exe -File {script}")
        lines = result.splitlines()
        messages = []
        for line in lines:
            if ":" in line:
                id_part, message_part = line.split(":", 1)
                messages.append([message_part.strip(), id_part.strip()])
        return messages
