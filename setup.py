from setuptools import setup, find_packages
import os

# Paths to the requirements.txt and Logicytics.version files
requirements_path = 'requirements.txt'
version_path = os.path.join('SYSTEM', 'Logicytics.version')

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
