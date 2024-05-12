# Logicytics: System Data Harvester

Welcome to **Logicytics** (Spelled Loj-ee-cit-iks), a powerful tool designed to harvest and collect a wide range of windows system data for forensics, it is a actively worked on project that uses python as its main language, its whole idea is it effectively gathers as much sensitive data as possible and outputs it into a ZIP file. This guide will help you get started with using DataVoyager effectively.

## Prerequisites

Before running Logicytics, it's recommended to first disable User Account Control (UAC) to ensure smooth operation. You can do this by running the `UACPY.py` script as an administrator in the Command Prompt (cmd). Here's how:

1. Open Command Prompt as an administrator. You can do this by searching for `cmd` in the Start menu, right-clicking on it, and selecting "Run as administrator".
2. Navigate to the directory where `UACPY.py` is located.
3. Execute the script by typing the following command and pressing Enter:

```powershell
python UACPY.py
```

Please note that this assumes you have Python installed on your system and that the `UACPY.py` script is located in the directory you navigate to in step 2. If Python is not installed or if you encounter any issues, you may need to install Python or adjust the command to point to your Python executable if it's not in your system's PATH.

It's also recommended to install all needed libraries, Here is how:

1. Open Command Prompt as an administrator. You can do this by searching for `cmd` in the Start menu, right-clicking on it, and selecting "Run as administrator".
2. Navigate to the directory where `requirements.txt` is located.
3. Execute the script by typing the following command and pressing Enter:

```cmd
pip install -r requirements.txt
```

## Running Logicytics

To run the main program, you need to execute `Logicytics.py` with administrative privileges (Note its not needed to run as admin, but half of it's functionality would be disabled). Follow these steps:

1. Open Command Prompt as an administrator.
2. Navigate to the directory where `Logicytics.py` is located.
3. Run the script by typing the following command and pressing Enter:

```cmd
python Logicytics.py
```

## Important Notes

- **Do Not Remove or Delete Any Folders or Files**: The integrity of the data collection process depends on the presence of all necessary files and folders. Removing or deleting any part of the Logicytics package could lead to errors or incomplete data collection.

- **Third-Party Tool's**: For those interested in additional functionality, you can explore more 3rd party software in the EXTRA tab. This tool offers advanced memory analysis capabilities and more features.

- **Initial Delay**: After starting Logicytics, you might not see any immediate feedback or activity for about 1 minute. This is normal and part of the data collection process.

- **Access Permissions**: The only files you should access after running Logicytics are the generated ZIP file and the `.md` log file (WIP). These files contain the collected data and log information, respectively and are found in the CODE sub-directory, you can freely move them anywhere.

- **Releases**: Don't download files from there, that is just some-sort of mini update log, download from the main branch, so No, old versions won't be saved here, and might not be supported.

- **Credits**: In the credits you will find many people, firms and projects that we took/used code/software from, these will explain what and who and why we did that, if you aare the creator of the project and dont want us to use your code, you are free to communicate with us so we can take your code down.


## Conclusion

Logicytics is a powerful tool for system data analysis. By following the instructions above, you can ensure a smooth and effective data collection process. Remember, the key to successful data harvesting is patience and adherence to the guidelines provided. Happy data mining!

And We are not responsible for any illegal usage of this product.
