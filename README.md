# Logicytics: System Data Harvester

Welcome to **Logicytics**, a powerful tool designed to harvest and collect a wide range of Windows system data for forensics. It's an actively developed project that primarily uses Python. Its goal is to gather as much sensitive data as possible and output it into a ZIP file. This guide will help you get started with using Logicytics effectively.

<div align="center">
    <a href="https://github.com/DefinetlyNotAI/Logicytics/issues"><img src="https://img.shields.io/github/issues/DefinetlyNotAI/Logicytics" alt="GitHub Issues"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/tags"><img src="https://img.shields.io/github/v/tag/DefinetlyNotAI/Logicytics" alt="GitHub Tag"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/t/DefinetlyNotAI/Logicytics" alt="GitHub Commit Activity"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/languages"><img src="https://img.shields.io/github/languages/count/DefinetlyNotAI/Logicytics" alt="GitHub Language Count"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/actions"><img src="https://img.shields.io/github/check-runs/DefinetlyNotAI/Logicytics/main" alt="GitHub Branch Check Runs"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/blob/master/LICENSE"><img src="https://img.shields.io/github/license/DefinetlyNotAI/Logicytics" alt="GitHub License"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics"><img src="https://img.shields.io/github/repo-size/DefinetlyNotAI/Logicytics" alt="GitHub Repo Size"></a>
</div>


### Agree to the ToS

Due to the use of third-party applications, they come with their own set of Terms of Service. It's mandatory to read the `!! Important!!.md` file located in the CODE/sys directory.

We also have our own ToS, it will prompt you when you first run Logicytics to agree to the ToS; don't worry as its small short and straightforward to read, and we don't attempt to trick you into selling your soul.

## Running Logicytics

To run the main program, you need to execute `Logicytics.py` with administrative privileges. Follow these steps:

1. Open Command Prompt as an administrator.
2. Navigate to the directory where `Logicytics.py` is located.
3. Run the script by typing the following command and pressing Enter:

```powershell
.\Logicytics.py
```

This will show the available command flags to use, if you want the default experience execute the following command `.\Logicytics.py --run`

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
