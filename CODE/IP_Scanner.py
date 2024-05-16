import socket
import subprocess
import os
import colorlog

# Configure colorlog
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)  # Set the log level
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def run_nmap_scan(target_ip=None):
    logger.info("Collecting IP...")
    if target_ip is None:
        target_ip = get_local_ip()

    # Get the current username
    username = os.getlogin()
    logger.info(f"Scanning {target_ip} from {username}... This might take a while... [~20 seconds]")

    # Construct the filename using the username
    filename = f"{username}_IP_Data.txt"

    # Use Nmap for a more thorough scan
    nmap_command = ['nmap', '-v', '-sV', '-O', '-Pn', '-T4', target_ip]
    process = subprocess.Popen(nmap_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, _ = process.communicate()

    output_str = output.decode().strip()

    # Write the output to a file
    with open(filename, 'w') as file:
        file.write(output_str)

    logger.info(f"Output written to {filename}")


# Example usage
logger.info("Setting up...")
run_nmap_scan()
