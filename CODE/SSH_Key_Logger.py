import getpass

from local_libraries.Setups import *


def copy_ssh_files_to_txt():
    """
    This function copies the contents of all files in the.ssh directory to a single SSH_DATA.txt file.
    """
    # Dynamically get the username of the current user
    username = getpass.getuser()

    # Construct the path to the.ssh directory using the username
    ssh_directory = rf"C:\Users\{username}\.ssh"

    # Debugging: Log the constructed path
    logger.info(f"Constructed path to .ssh directory: {ssh_directory}")

    # Check if the directory exists
    if not os.path.exists(ssh_directory):
        logger.warning(
            "The specified directory does not exist, most likely due to no SSH keys"
        )
    else:
        # Get the current working directory
        cwd = os.getcwd()

        # Construct the path for the SSH_DATA.txt file in the current working directory
        outfile_path = os.path.join(cwd, "SSH_DATA.txt")

        # Debugging: Log the constructed path for the output file
        logger.info(f"Constructed path for SSH_DATA.txt: {outfile_path}")

        # Open (or create) the SSH_DATA.txt file in write mode
        try:
            with open(outfile_path, "w") as outfile:
                # List all files in the specified directory
                for filename in os.listdir(ssh_directory):
                    # Construct the full file path
                    file_path = os.path.join(ssh_directory, filename)

                    # Check if it is a file (not a directory)
                    if os.path.isfile(file_path):
                        # Open the file in read mode
                        with open(file_path, "r") as infile:
                            # Write the content of the file to SSH_DATA.txt
                            outfile.write(f"--- Start of {filename} ---\n")
                            outfile.write(infile.read())
                            outfile.write(f"\n--- End of {filename} ---\n")

                logger.info(
                    "All SSH files have been successfully copied to SSH_DATA.txt."
                )
        except Exception as e:
            logger.error(f"An error occurred while processing the files: {e}")


# Example usage
if __name__ == "__main__":
    copy_ssh_files_to_txt()
