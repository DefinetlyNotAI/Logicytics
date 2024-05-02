# DataVoyager: System Data Harvester

Welcome to **DataVoyager**, a powerful tool designed to harvest and collect a wide range of windows system data for forensics. This guide will help you get started with using DataVoyager effectively.

## Prerequisites

Before running DataVoyager, it's recommended to first disable User Account Control (UAC) to ensure smooth operation. You can do this by running the `UACX.ps1` script as an administrator in the Command Prompt (cmd). Here's how:

1. Open Command Prompt as an administrator. You can do this by searching for `cmd` in the Start menu, right-clicking on it, and selecting "Run as administrator".
2. Navigate to the directory where `UACX.ps1` is located.
3. Execute the script by typing the following command and pressing Enter:

```powershell
powershell.exe -ExecutionPolicy Bypass -File UACX.ps1
```

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

Make sure a password is available to the admin account, this is because this will ask for a password and not allow you to enter nothing.
```cmd
runas /user:Administrator miner.py
```

## Important Notes

- **Do Not Remove or Delete Any Folders or Files**: The integrity of the data collection process depends on the presence of all necessary files and folders. Removing or deleting any part of the DataVoyager package could lead to errors or incomplete data collection.

- **Memory Dumper Tool**: For those interested in additional functionality, you can explore the Memory Dumper tool located in the Memory Dumper folder. This tool offers advanced memory analysis capabilities.

- **Initial Delay**: After starting DataVoyager, you might not see any immediate feedback or activity for about 5 minutes. This is normal and part of the data collection process.

- **Access Permissions**: The only files you should access after running DataVoyager are the generated ZIP file and the `.md` log file. These files contain the collected data and log information, respectively.

## Conclusion

DataVoyager is a powerful tool for system data analysis. By following the instructions above, you can ensure a smooth and effective data collection process. Remember, the key to successful data harvesting is patience and adherence to the guidelines provided. Happy data mining!
