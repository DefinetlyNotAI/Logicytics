# DataVoyager: System Data Harvester

Welcome to **DataVoyager**, a powerful tool designed to harvest and collect a wide range of windows system data for forensics. This guide will help you get started with using DataVoyager effectively.

## Prerequisites

Before running DataVoyager, it's recommended to first disable User Account Control (UAC) to ensure smooth operation. You can do this by running the `UACX.py` script as an administrator in the Command Prompt (cmd). Here's how:

1. Open Command Prompt as an administrator. You can do this by searching for `cmd` in the Start menu, right-clicking on it, and selecting "Run as administrator".
2. Navigate to the directory where `UACX.py` is located.
3. Execute the script by typing the following command and pressing Enter:

```powershell
python UACX.py
```

Please note that this assumes you have Python installed on your system and that the `UACX.py` script is located in the directory you navigate to in step 2. If Python is not installed or if you encounter any issues, you may need to install Python or adjust the command to point to your Python executable if it's not in your system's PATH.

It's also recommended to install all needed libraries, Here is how:

1. Open Command Prompt as an administrator. You can do this by searching for `cmd` in the Start menu, right-clicking on it, and selecting "Run as administrator".
2. Navigate to the directory where `requirements.txt` is located.
3. Execute the script by typing the following command and pressing Enter:

```cmd
pip install -r requirements.txt
```

## Running DataVoyager

To run the main program, you need to execute `miner.py` with administrative privileges. Follow these steps:

1. Open Command Prompt as an administrator.
2. Navigate to the directory where `miner.py` is located.
3. Run the script by typing the following command and pressing Enter:

```cmd
python miner.py
```

## Important Notes

- **Privilege Escalation**: I will not include any sort of exploit to privelage access at the moment, this is to ensure the tool is used within legal limits, Maybe in the future a sort of signing legal mechanism may be implemented, but that is later
  
- **Do Not Remove or Delete Any Folders or Files**: The integrity of the data collection process depends on the presence of all necessary files and folders. Removing or deleting any part of the DataVoyager package could lead to errors or incomplete data collection.

- **Memory Dumper Tool**: For those interested in additional functionality, you can explore the Memory Dumper tool located in the Memory Dumper folder. This tool offers advanced memory analysis capabilities.

- **Initial Delay**: After starting DataVoyager, you might not see any immediate feedback or activity for about 5 minutes. This is normal and part of the data collection process.

- **Access Permissions**: The only files you should access after running DataVoyager are the generated ZIP file and the `.md` log file. These files contain the collected data and log information, respectively.

## Conclusion

DataVoyager is a powerful tool for system data analysis. By following the instructions above, you can ensure a smooth and effective data collection process. Remember, the key to successful data harvesting is patience and adherence to the guidelines provided. Happy data mining!
