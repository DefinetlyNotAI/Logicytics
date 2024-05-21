# Logicytics: System Data Harvester

Welcome to **Logicytics**, a powerful tool designed to harvest and collect a wide range of Windows system data for forensics. It's an actively developed project that primarily uses Python. Its goal is to gather as much sensitive data as possible and output it into a ZIP file. This guide will help you get started with using Logicytics effectively.

## Prerequisites

Before running Logicytics, please follow these steps:

### Disable User Account Control (UAC)

To ensure smooth operation and enable cmd, disable UAC by running the `UACPY.py` script as an administrator in the Command Prompt (cmd).

#### Steps:

1. **Open Command Prompt as Administrator:**
   - Search for `cmd` in the Start menu.
   - Right-click on it and select "Run as administrator."

2. **Navigate to the Script Directory:**
   - Use the `cd` command to change directories to where `UACPY.py` is located.

3. **Execute the Script:**
   - Type `.\UACPY.py` and press Enter.

**Note:** This requires Python to be installed on your system. The script should be in the directory you navigate to in step 2. If Python isn't installed or there are issues, consider installing Python or adjusting the command to point to your Python executable if it's not in your system's PATH.

### Install Required Libraries

Install the necessary libraries by executing the following commands in the Command Prompt as an administrator:

1. **Open Command Prompt as Administrator:**
   - Follow the same method as above.

2. **Navigate to the Requirements File Location:**
   - Change directories to where `requirements.txt` is located using the `cd` command.

3. **Install Libraries:**
   - Type `pip install -r requirements.txt` and press Enter.

### Cripple Windows Defender

To improve performance, it's recommended to temporarily disable Windows Defender. Run the following command as an administrator:

```cmd
.\Window_Defender_Crippler.bat
```

**Important:** Running this script again will reinstall the signatures. After completing your tasks, re-run this file to restore Windows Defender's protection.

### Agree to the ToS

Due to the use of third-party applications, they come with their own set of Terms of Service. It's mandatory to read the `!! Important!!.md` file located in the CODE/sys directory.

We also have our own ToS, you will be prompted by them when you first run Logicytics.

## Running Logicytics

To run the main program, you need to execute `Logicytics.py` with administrative privileges. Follow these steps:

1. Open Command Prompt as an administrator.
2. Navigate to the directory where `Logicytics.py` is located.
3. Run the script by typing the following command and pressing Enter:

```powershell
.\Logicytics.py
```

## Running Debugger

To run the debugger program, you need to execute `Debug.py` with administrative privileges. Follow these steps:

1. Open Command Prompt as an administrator.
2. Navigate to the directory where `DebugBeta.py` is located (will be in the CODE directory).
3. Run the script by typing the following command and pressing Enter:
4. You will receive DEBUG.md, which contains the file required for reporting bugs.

```powershell
.\Debug.py
```

Ensure the `.structure` file is present. If you don't have it, download the `.structure` file found in the SYSTEM directory of this repo.

## Important Notes

- **Do Not Remove or Delete Any Folders or Files:** The integrity of the data collection process depends on the presence of all necessary files and folders. Removing or deleting any part of the Logicytics package could lead to errors or incomplete data collection.

- **Third-Party Tools:** For those interested in additional functionality, you can explore more third-party software in the EXTRA tab. This tool offers advanced memory analysis capabilities and more features.

- **Access Permissions:** The only files you should access after running Logicytics are the generated ZIP file and the `.md` log file (WIP). These files contain the collected data and log information, respectively, and are found in the CODE subdirectory; you can freely move them anywhere.

- **Releases:** Don't download files from there; that is just some sort of mini-update log. Download from the main branch; old versions won't be saved here and might not be supported.

- **Credits:** In the credits, you will find many people, firms, and projects whose code/software we used. If you are the creator of the project and don't want us to use your code, feel free to contact us, so we can remove it.

- **Explore:** Check all the files and ReadMe to understand how and what the project does.

## Conclusion

Logicytics is a powerful tool for system data analysis. By following the instructions above, you can ensure a smooth and effective data collection process. Remember, the key to successful data harvesting is patience and adherence to the guidelines provided. Happy data mining!

We are not responsible for any illegal usage of this product.
