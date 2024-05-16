import socket
import subprocess
import os


def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def run_nmap_scan(target_ip=None):
    print("Collecting IP...")
    if target_ip is None:
        target_ip = get_local_ip()

    # Get the current username
    username = os.getlogin()
    print(f"Scanning {target_ip} from {username}... This might take a while... [~20 seconds]")

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

    print(f"Output written to {filename}")


# Example usage
print("Setting up...")
run_nmap_scan()
