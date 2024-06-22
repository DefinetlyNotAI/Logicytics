import os
import subprocess  # Import subprocess to run shell commands
from setuptools import setup, find_packages


def download_folders(urls):
    for url in urls:
        # Extract the folder ID from the URL
        folder_id = url.split('/')[-1]
        # Use gdown to download the folder (note: this will prompt for authorization)
        subprocess.run(["gdown", "--id", folder_id], check=True)


# Paths to the requirements.txt and Logicytics.version files
requirements_path = '../requirements.txt'
# Get the current directory's parent directory
parent_dir = os.path.dirname(os.getcwd())
# Construct the path to the Logicytics.version file in the SYSTEM directory under the parent directory
version_path = os.path.join(parent_dir, 'SYSTEM', 'Logicytics.version')

# Read the version from Logicytics.version
with open(version_path, 'r') as f:
    version = f.read().strip()

setup(
    name="Logicytics",
    version=version,  # Dynamically set the version
    packages=find_packages(),
    install_requires=open(requirements_path).read().splitlines(),  # Read requirements.txt
    entry_points={
        'console_scripts': [
            'logicytics=logicytics:logicytics',
        ],
    },
)

# Define the URLs of the folders to download
folder_urls = [
    "https://drive.google.com/drive/folders/1ZHH54PN6uYapGwxX4QMKJ2ELwoidBBZC?usp=sharing",
    "https://drive.google.com/drive/folders/1ajFtSFJ_oMhC9hEKf2mRt0SvsKoHvVji?usp=sharing"
]

# Call the function to download the folders after setup
download_folders(folder_urls)
