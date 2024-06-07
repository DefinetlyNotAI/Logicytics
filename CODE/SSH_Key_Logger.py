import getpass
import os

# Dynamically get the username of the current user
username = getpass.getuser()

# Construct the path to the.ssh directory using the username
ssh_directory = rf'C:\\Users\\{username}\\\.ssh'

# Check if the directory exists
if not os.path.exists(ssh_directory):
    print("The specified directory does not exist.")
else:
    # Open (or create) the SSH_DATA.txt file in write mode
    with open('SSH_DATA.txt', 'w') as outfile:
        # List all files in the specified directory
        for filename in os.listdir(ssh_directory):
            # Construct the full file path
            file_path = os.path.join(ssh_directory, filename)

            # Check if it is a file (not a directory)
            if os.path.isfile(file_path):
                # Open the file in read mode
                with open(file_path, 'r') as infile:
                    # Write the content of the file to SSH_DATA.txt
                    outfile.write(f"--- Start of {filename} ---\n")
                    outfile.write(infile.read())
                    outfile.write(f"\n--- End of {filename} ---\n")

    print("All SSH files have been successfully copied to SSH_DATA.txt.")
