# Logicytics: System Data Harvester 📎

Welcome to Logicytics 🌐,
a cutting-edge tool
designed to meticulously harvest and collect a vast array of Windows system data for forensic analysis.
Crafted with Python 🐍,
it's an actively developed project that is
aimed at gathering as much sensitive data as possible and packaging it neatly into a ZIP file 📦.
This comprehensive guide is here to equip you with everything you need to use Logicytics effectively.

<div align="center">
    <a href="https://github.com/DefinetlyNotAI/Logicytics/issues"><img src="https://img.shields.io/github/issues/DefinetlyNotAI/Logicytics" alt="GitHub Issues"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/tags"><img src="https://img.shields.io/github/v/tag/DefinetlyNotAI/Logicytics" alt="GitHub Tag"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/t/DefinetlyNotAI/Logicytics" alt="GitHub Commit Activity"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/languages"><img src="https://img.shields.io/github/languages/count/DefinetlyNotAI/Logicytics" alt="GitHub Language Count"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics/actions"><img src="https://img.shields.io/github/check-runs/DefinetlyNotAI/Logicytics/main" alt="GitHub Branch Check Runs"></a>
    <a href="https://github.com/DefinetlyNotAI/Logicytics"><img src="https://img.shields.io/github/repo-size/DefinetlyNotAI/Logicytics" alt="GitHub Repo Size"></a>
</div>

## ❤️ Support Me ❤️

Please consider buying me a coffee or sponsoring me in GitHub sponsor,
I am saving for my college funds in MIT, and I need your help!
Supporters will be placed in the Credits ❤️

## 🛠️ Installation and Setup 🛠️

### Prerequisites

Ensure your system meets these requirements:

- Windows 10 or later.
- Administrative rights.
- Python installed (preferable version 3.11).

### Step-by-Step Installation

Optionally, use the `exe` installer to download and install Logicytics and all requirements with a simple GUI interface.

1. **Clone the Repository**: Use Git to clone Logicytics to your local machine. Open Command Prompt as an administrator
   and run:
   ```powershell
   git clone https://github.com/DefinetlyNotAI/Logicytics.git
   ```
2. **Navigate to the Project Directory**: Change your current directory to the cloned Logicytics folder:
   ```powershell
   cd Logicytics
   ```

3. **Setup Logicytics**: Navigate to the `SETUP` folder and run the `setup.py` script using the following command,
   This will install the required dependencies and create any necessary files for full functionality.
   ```powershell
   pip install -e .
   ```

4. **Run Logicytics**: Navigate to the `CODE` folder and run `./Logicytics.py` more info below.

### Basic Usage

1. **CLI**: The most preferred method, all in your command line! More feedback can be given, to use it just open `CMD`
   in the `CODE` directory, and run `./Logicytics.py`

2. **GUI**: Still in beta, GUI is not recommended unless you are unfamiliar with CLI, the GUI automatically constructs
   your command and executes it from a basic window. To use this feature, go to the `CODE` directory and run
   the `GUI.py` file, or type in the terminal `./GUI.py`.

## 🔧 Configuration 🔧

Logicytics offers extensive customization options through flags while running.

These flags allow you to:

- Specify which types of files to collect data.
- Exclude certain files/data from the data collection process.
- Adjust logging levels for detailed insights.

More info about the flags on the wiki.

## 🚀 Advanced Usage 🚀

### Custom Scripts

Extend Logicytics' functionality by creating custom Python scripts.
Place these scripts in the `CODE` directory.
Logicytics will automatically execute these scripts during the data collection process,
enabling tailored data extraction (When using the `--mods` flag).

## 🛑 Troubleshooting 🛑

### Common Pitfalls

- **Permission Denied**: Ensure you're running Logicytics with administrative privileges.
- **Incomplete Data Collection**: Verify all necessary files and folders are intact and unmodified.
- **Update Issues**: Use the `--update` flag to fetch the latest version of Logicytics.
- **Recovery**: Use the `--backup` and `--restore` to keep a history of intact files in case of errors.

### Support Resources

Consult the `.md` log file in the `ACCESS/LOGS` directory for detailed error logs.
Engage with the community through GitHub issues for assistance and feedback.

## 📊 Data Analysis 📊

Once Logicytics has completed its data collection,
you'll find the results packaged neatly in a ZIP file within the `ACCESS/DATA` directory.
This data can be analyzed using various tools and techniques, depending on your needs.
Whether you're conducting forensic investigations, auditing system health,
or analyzing performance metrics, Logicytics provides a solid foundation for your analysis.

## 🌟 Conclusion 🌟

By exploring the depths of Logicytics, you've gained a deeper understanding of its capabilities,
configuration options, and advanced features.
This tool is a powerful asset in your arsenal for system data analysis,
offering flexibility, customization, and ease of use.
Remember, the key to unlocking its full potential lies in experimentation and continuous learning.
Happy data mining 🎯

This expanded guide aims to provide a thorough understanding of Logicytics,
covering everything from installation and setup to advanced usage and troubleshooting.
With this knowledge, you're well-equipped to utilize Logicytics to its fullest extent,
enhancing your ability to analyze and understand system data.
