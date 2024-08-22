import os
import shutil

def backup_ssh_keys_and_config():
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the path to the SSH directory
    ssh_folder = os.path.join(os.environ['USERPROFILE'], '.ssh')

    # Define the destination directory as the current working directory
    destination_dir = current_dir

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Define source and destination directories
    source_dir = ssh_folder
    destination_dir = os.path.join(current_dir, 'ssh_backup')  # Use a subdirectory named 'ssh_backup' in the current directory

    # Copy SSH keys and config
    try:
        shutil.copytree(source_dir, destination_dir)
        print("SSH keys and configuration backed up successfully.")
    except Exception as e:
        print(f"Failed to back up SSH keys and configuration: {e}")

if __name__ == "__main__":
    backup_ssh_keys_and_config()
