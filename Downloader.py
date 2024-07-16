import os
import subprocess
import shutil
from pathlib import Path

# Define paths relative to the script's location
script_dir = Path(__file__).resolve().parent
script_path = os.path.realpath(__file__)

# Function to check if Python is installed
def is_python_installed():
    try:
        import sys
        return sys.version_info >= (3, 11)
    except ImportError:
        print("Python is not installed. Installing now...")
        install_python()
        is_python_installed()
        

# Function to install Python
def install_python():
    # Your installation logic here
    pass  # Placeholder for the installation logic

# Function to clone the repository if it doesn't already exist
def clone_repo_if_not_exists(repo_url):
    repo_name = "Logicytics"
    # Check if the repository directory already exists in the current working directory
    if not any(path.name == repo_name and path.is_dir() for path in script_dir.iterdir()):
        subprocess.run(["git", "clone", repo_url], cwd=script_dir)
    else:
        print(f"The repository {repo_name} already exists in the current directory. Skipping clone.")

# Main function
def main():
    # Check if Python is installed
    is_python_installed():

    # Clone the repository into the current directory where Downloader.py is located
    clone_repo_if_not_exists("https://github.com/DefinetlyNotAI/Logicytics.git")
    
    # Assuming the SETUP directory exists within the cloned repository
    setup_dir = script_dir / "Logicytics" / "SETUP"
    if setup_dir.exists():
        os.chdir(setup_dir)
        
        # Run pip install
        result = subprocess.run(["pip", "install", "-e", "."], capture_output=True, text=True)
        with open("Download.log", "w") as log_file:
            log_file.write(result.stdout)
        
        # Prepare the logs directory
        logs_dir = script_dir / "Logicytics" / "ACCESS" / "LOGS"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Move Download.log to LOGS directory
        shutil.move(setup_dir / "Download.log", logs_dir)
        
        # Delete this file
        os.system(f'del "{script_path}"')
    else:
        print(f"The SETUP directory does not exist at {setup_dir}. Cannot proceed.")

if __name__ == "__main__":
    main()
