import os

from setuptools import setup, find_packages

# Paths to the requirements.txt and Logicytics.version files
requirements_path = "../requirements.txt"
# Get the current directory's parent directory
parent_dir = os.path.dirname(os.getcwd())
# Construct the path to the Logicytics.version file in the SYSTEM directory under the parent directory
version_path = os.path.join(parent_dir, "SYSTEM", "Logicytics.version")

# Read the version from Logicytics.version
with open(version_path, "r") as f:
    version = f.read().strip()

setup(
    name="Logicytics",
    version=version,  # Dynamically set the version
    packages=find_packages(),
    install_requires=open(requirements_path)
    .read()
    .splitlines(),  # Read requirements.txt
    entry_points={
        "console_scripts": [
            "logicytics=logicytics:logicytics",
        ],
    },
)
