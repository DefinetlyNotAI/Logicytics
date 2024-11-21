# Logicytics: System Data Harvester 📎

Logicytics is a cutting-edge tool designed to 
meticulously harvest and collect a vast array of Windows system data for forensic analysis.
Crafted with Python 🐍, it's an actively developed project that is dedicated
to gathering as much sensitive data as possible and packaging it neatly into a ZIP file 📦.
This comprehensive guide is here to equip you with everything you need to use Logicytics effectively.

<div style="text-align:center;" align="center">
    <a href="https://github.com/DefinetlyNotAI/Logicytics/issues"><img src="https://img.shields.io/github/issues/DefinetlyNotAI/Logicytics" alt="GitHub Issues"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/tags"><img src="https://img.shields.io/github/v/tag/DefinetlyNotAI/Logicytics" alt="GitHub Tag"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/t/DefinetlyNotAI/Logicytics" alt="GitHub Commit Activity"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/languages"><img src="https://img.shields.io/github/languages/count/DefinetlyNotAI/Logicytics" alt="GitHub Language Count"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/actions"><img src="https://img.shields.io/github/check-runs/DefinetlyNotAI/Logicytics/main" alt="GitHub Branch Check Runs"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics"><img src="https://img.shields.io/github/repo-size/DefinetlyNotAI/Logicytics" alt="GitHub Repo Size"></a>
</div>
<div style="text-align:center;" align="center">
    <a href="https://www.codefactor.io/repository/github/definetlynotai/logicytics"><img src="https://www.codefactor.io/repository/github/definetlynotai/logicytics/badge" alt="GitHub Repo CodeFactor Rating"></a>
    <a href="https://codeclimate.com/github/DefinetlyNotAI/Logicytics/maintainability"><img src="https://api.codeclimate.com/v1/badges/ae2c436af07d00aabf00/maintainability"  alt="GitHub Repo CodeClimate Rating"/></a>
    <a href="https://api.securityscorecards.dev/projects/github.com/DefinetlyNotAI/Logicytics"><img src="https://api.securityscorecards.dev/projects/github.com/DefinetlyNotAI/Logicytics/badge"  alt="OpenSSF Best Practices Score"/></a>
    <a href="https://www.bestpractices.dev/projects/9451"><img src="https://www.bestpractices.dev/projects/9451/badge" alt="OpenSSF Best Practices Badge"></a>
</div>

> [!CAUTION]
> By using this software, you agree to the license, and agree that you hold responsibility of how you use and modify the code.

## 🛠️ Installation and Setup 🛠️

To install and setup Logicytics, follow these steps:

1. **Install Python**: If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/).

2. **Install Dependencies**: Logicytics requires Python modules. You can install all the required modules by running the following command in your terminal: `pip install -r requirements.txt`

3. **Run Logicytics**: To run Logicytics, simply run the following command in your terminal: `python Logicytics.py -h` - This opens a help menu.

### Prerequisites

- **Python**: The project requires Python 3.8 or higher. You can download Python from the [official website](https://www.python.org/downloads/).

- **Dependencies**: The project requires certain Python modules to be installed. You can install all the required modules by running the following command in your terminal: `pip install -r requirements.txt`.

- **Administrative Privileges**: To be able to run the program using certain features of the project, like registry modification, you must run the program with administrative privileges.

- **System Requirements**: The project has been tested on Windows 10 and 11. It will not work on other operating systems.

- **Knowledge of Command Line**: The project uses command line options for the user to interact with the program. It is recommended to have a basic understanding of command line options.

> [!IMPORTANT]
> You may create a `.sys.ignore` file in the `CODE/SysInternal_Suite` directory to not extract the exe binaries from the ZIP file (This is done for the OpenSSF score and to discourage binaries being used without source code), if the `.sys.ignore` file is not found, it will auto extract the binaries and run them using `Logicytics`. For more details on these binaries, go [here](https://learn.microsoft.com/en-us/sysinternals/downloads/sysinternals-suite) - For you weary cautious internet crusaders, you can view the [source code here](https://github.com/MicrosoftDocs/sysinternals) and compare hashes and perform your audits.

## Step-by-Step Installation and Usage

1) Install Python 🐍
If you don't have Python installed, you can download it from the <a href="https://www.python.org/downloads/">official website</a>.
Make sure to select the option to "Add Python to PATH" during installation.

2) Install Dependencies 📦
Logicytics requires Python modules. You can install all the required modules by running the following command in your terminal:
`pip install -r requirements.txt`

3) Run Logicytics 🚀
To run Logicytics, simply run the following command in your terminal:
<code>python Logicytics.py -h</code>
This opens a help menu.

4) Run the Program 👾
Once you have run the program, you can run the program with the following command:
`python Logicytics.py -h`
Replace the flags with the ones you want to use.
you must have admin privileges while running!

> [!TIP]
> Although it's really recommended to use admin, by setting debug in the config.json to true, you can bypass this requirement

6) Wait for magic to happen 🧙‍♀️
Logicytics will now run and gather data according to the flags you used.

7) Enjoy the gathered data 🎉
Once the program has finished running, you can find the gathered data in the "ACCESS/DATA" folder. Both Zip and Hash will be found there.

> [!NOTE]
> All Zips and Hashes follow a conventional naming mechanism that goes as follows
> `Logicytics_{CODE-or-MODS}_{Flag-Used}_{Date-And-Time}.zip`

9) Share the love ❤️
If you like Logicytics, please consider sharing it with others or spreading the word about it.

10) Contribute to the project 👥
If you have an idea or want to contribute to the project, you can submit an issue or PR on the <a href="https://github.com/DefinetlyNotAI/Logicytics">GitHub repository</a>.


### Basic Usage

After running and successfully collecting data, you may traverse the ACCESS directory as much as you like,
Remove add and delete files, it's the safe directory where your backups, hashes, data zips and logs are found.

> [!TIP]
> Watch this [video](https://www.youtube.com/watch?v=XVTBmdTQqOs) to see a real life demo of Logicytics (Although the tools and interface may be changed as it's an older version)

## 🔧 Configuration 🔧

Logicytics uses a config.json file to store configurations. The config.json is located in the CODE directory.

The config.json file is a JSON file that contains the following information:

```json
{
    "Log Level Debug?": "boolean - True or False",
    "VERSION": "Version number x.x.x  format",
    "CURRENT_FILES": [
        "Files Seperated by commas in strings"
    ]
}
```

The config.json file is used to store the DEBUG flag bool, the VERSION, and the CURRENT_FILES.

> [!TIP]
> CURRENT_FILES is an array of strings that contains the names of the files you have, 
> this is used to later check for corruption or bugs.
> VERSION is the version of the project, used to check and pull for updates.

## 🚀 Advanced Usage 🚀

### Mods

Mods are special files that are run with the `--modded` flag.
These files are essentially scripts that are run after the main Logicytics.py script is run
and the verified scripts are run.

They are used to add extra functionality to the script. 
They are located in the `MODS` directory. In order to make a mod, 
you need to create a python file with the `.py` extension or any of the supported extensions `.exe .ps1 .bat`
in the `MODS` directory. 

These file will be run after the main script is run. 
When making a mod, you should avoid acting based on other files directly, 
as this can cause conflicts with the data harvesting. 
Instead, you should use the `Logicytics.py` file and other scripts as a reference 
for how to add features to the script.

The `--modded` flag is used to run all files in the `MODS` directory. 
This flag is not needed for other files in the `CODE` directory to run,
but it is needed for mods to run. 

The `--modded` flag can also be used to run custom scripts.
If you want to run a custom script with the `--modded` flag, 
you can add the script to the `MODS` directory, and it will be run with the `--modded` flag.

To check all the mods and how to make your own, you can check the `Logicytics.py` file and the Wiki.
Also refer to the contributing.md for more info

## 🛑 Troubleshooting 🛑

If you are having issues, here are some troubleshooting tips:

Some errors may not necessarily mean the script is at fault, 
but other OS related faults like files not existing,
or files not being modified, or files not being created.

Some tips are:
- Check if the script is running as admin and not in a VM
- Check if the script has the correct permissions and correct dependencies to run
- Check if the script is not being blocked by a firewall or antivirus or by a VPN or proxy
- Check if the script is not being blocked by any other software or service

If those don't work attempt:
- Try running the script with powershell instead of cmd, or vice versa
- Try running the script in a different directory, computer or python version above 3.8
  - Note: The version used to develop, test and run the script is 3.11
- Try running the `--debug` flag and check the logs

### Support Resources

Check out the [GitHub wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki) for help.

## 📊 Data Analysis 📊

## Data Extraction

Logicytics extracts a wide range of data points on a Windows system.

Here are some of the data points that Logicytics extracts:

> [!IMPORTANT]
> Don't recreate the scripts/ideas below as then it's a waste of time for you, unless the Side-note on the script says otherwise, you can however contribute to the script itself.

> [!TIP]
> You can check out future plans [here](PLANS.md), you can contribute these plans if you have no idea's on what to contribute!

| File Name                | About                                                                                               | Important Note           |
|--------------------------|-----------------------------------------------------------------------------------------------------|--------------------------|
| browser_miner.ps1        | Mines all data related to browsers                                                                  | Would love to be updated |
| cmd_commands.py          | Gets data from driverquery, sysinfo, gpresult and more                                              |                          |
| log_miner.py             | Gets all logs from the Windows device                                                               |                          |
| media_backup.py          | Gets all media of the device in a neat folder                                                       | Would love to be updated |
| netadapter.ps1           | Runs Get-NetAdapter Command with many flags                                                         |                          |
| property_scraper.ps1     | Gets all the windows properties                                                                     |                          |
| registry.py              | Backups the registry                                                                                |                          |
| sensitive_data_miner.py  | Copies all files that can be considered sensitive in a neat folder, very slow and clunky            |                          |
| ssh_miner.py             | Gets as much ssh private data as possible                                                           |                          |
| sys_internal.py          | Attempts to use the Sys_Internal Suite from microsoft                                               |                          |
| tasklist.py              | Gets all running tasks, PID and info/data                                                           |                          |
| tree.bat                 | Runs and logs the tree.bat command                                                                  |                          |
| window_feature_miner.ps1 | Logs all the windows features enabled                                                               |                          |
| wmic.py                  | Logs and runs many wmic commands to gain sensitive data and information                             |                          |
| wifi_stealer.py          | Gets the SSID and Password of all saved Wi-Fi                                                       |                          |
| dir_list.py              | Produces a txt on every single file on the device, very slow and clunky - useful for depth scanning |                          |

This is not an exhaustive list, 
but it should give you a good idea of what data Logicytics is capable of extracting.

> [!NOTE]
> **Any file with `_` is not counted here, do note they may range from custom libraries to special files/wrappers**

### Want to create your own mod?

Check out the [contributing guidlines](CONTRIBUTING.md) file for more info

### Want More?

If there is a specific piece of data that you would like to see extracted by Logicytics,
please let us know. We are constantly working to improve the project and adding new features.

Other than mods, some prefixed tools are in the `EXTRA` directory, 
use the `--extra` flag to traverse these special tools

### Want to create your own mod?

Check out the [contributing guidlines](CONTRIBUTING.md) file for more info, as well as the [wiki guidelines](https://github.com/DefinetlyNotAI/Logicytics/wiki/5-Contribution-Guidelines) for more info

> [!IMPORTANT]
> Always adhere to the [coding standards](https://github.com/DefinetlyNotAI/Logicytics/wiki/6-Coding-Standards) of Logicytics!

## 🌟 Conclusion 🌟

Logicytics is a powerful tool that can extract a wide variety of data from a Windows system.
With its ability to extract data from various sources, Logicytics can be used for a variety of purposes,
from forensics to system information gathering. 
Its ability to extract data from various sources makes it a valuable tool
for any Windows system administrator or forensic investigator.

> [!CAUTION]
> Please remember that extracting data from a system without proper authorization is illegal and unethical.
> Always obtain proper authorization before extracting any data from a system.

## ❤️ Support Me ❤️

Please consider buying me a coffee or sponsoring me in GitHub sponsor,
I am saving for my college funds in MIT, and I need your help!
Supporters will be placed in the Credits ❤️

### 🔗 Links

- [Project's Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki)
- [Project's Future](PLANS.md)
- [Project's License](LICENSE)

### License

- [Developer Certificate of Origin](DCO.md)
- [Our License](LICENSE)
