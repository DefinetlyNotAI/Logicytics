import os
import socket
import time

import psutil

from logicytics import log, Execute


class NetworkInfo:
    """
    A class to gather and save various network-related information.
    """

    @log.function
    def get(self):
        """
        Gathers and saves various network-related information by calling multiple internal methods.
        """
        try:
            self.__fetch_network_io_stats()
            self.__fetch_network_connections()
            self.__fetch_network_interface_addresses()
            self.__fetch_network_interface_stats()
            self.__execute_external_network_command()
            self.__fetch_network_connections_with_process_info()
            self.__measure_network_bandwidth_usage()
            self.__fetch_hostname_and_ip()
        except Exception as e:
            log.error(f"Error getting network info: {e}, Type: {type(e).__name__}")

    @staticmethod
    def __save_data(filename: str, data: str, father_dir_name: str = "network_data"):
        """
        Saves the given data to a file.

        :param filename: The name of the file to save the data in.
        :param data: The data to be saved.
        :param father_dir_name: The directory to save the file in. Defaults to "network_data".
        """
        os.makedirs(father_dir_name, exist_ok=True)
        try:
            with open(os.path.join(father_dir_name, filename), "w") as f:
                f.write(data)
        except IOError as e:
            log.error(f"Failed to save {filename}: {e}")

    def __fetch_network_io_stats(self):
        """
        Fetches and saves network I/O statistics for each network interface.
        """
        log.debug("Fetching network interface stats...")
        net_io = psutil.net_io_counters(pernic=True)
        net_io_data = ""
        for iface, stats in net_io.items():
            net_io_data += f"Interface: {iface}\n"
            net_io_data += f"Bytes Sent: {stats.bytes_sent}, Bytes Received: {stats.bytes_recv}\n"
            net_io_data += f"Packets Sent: {stats.packets_sent}, Packets Received: {stats.packets_recv}\n"
            net_io_data += f"Errors In: {stats.errin}, Errors Out: {stats.errout}\n"
            net_io_data += f"Dropped In: {stats.dropin}, Dropped Out: {stats.dropout}\n\n"
        self.__save_data("network_io.txt", net_io_data)
        log.info("Network IO stats saved.")

    def __fetch_network_connections(self):
        """
        Fetches and saves information about network connections.
        """
        log.debug("Fetching network connections...")
        connections = psutil.net_connections(kind='all')
        connections_data = ""
        for conn in connections:
            connections_data += f"Type: {conn.type}, Local: {conn.laddr}, Remote: {conn.raddr}, Status: {conn.status}\n"
        self.__save_data("network_connections.txt", connections_data)
        log.info("Network connections saved.")

    def __fetch_network_interface_addresses(self):
        """
        Fetches and saves network interface addresses.
        """
        log.debug("Fetching network interface addresses...")
        interfaces = psutil.net_if_addrs()
        interfaces_data = ""
        for iface, addrs in interfaces.items():
            for addr in addrs:
                interfaces_data += f"Interface: {iface}, Address: {addr.address}, Netmask: {addr.netmask}, Broadcast: {addr.broadcast}\n"
        self.__save_data("network_interfaces.txt", interfaces_data)
        log.info("Network interface addresses saved.")

    def __fetch_network_interface_stats(self):
        """
        Fetches and saves network interface statistics.
        """
        log.debug("Fetching network interface stats...")
        stats = psutil.net_if_stats()
        stats_data = ""
        for iface, stat in stats.items():
            stats_data += f"Interface: {iface}, Speed: {stat.speed}Mbps, Duplex: {stat.duplex}, Up: {stat.isup}\n"
        self.__save_data("network_stats.txt", stats_data)
        log.info("Network interface stats saved.")

    def __execute_external_network_command(self):
        """
        Executes an external network command and saves the output.
        """
        log.debug("Executing external network command...")
        result = Execute.command("ipconfig")
        self.__save_data("network_command_output.txt", result)
        log.info("Network command output saved.")

    def __fetch_network_connections_with_process_info(self):
        """
        Fetches and saves network connections along with associated process information.
        """
        log.debug("Fetching network connections with process info...")
        connections_data = ""
        for conn in psutil.net_connections(kind='all'):
            pid = conn.pid if conn.pid else "N/A"
            proc_name = "Unknown"
            if pid != "N/A":
                try:
                    proc_name = psutil.Process(pid).name()
                except psutil.NoSuchProcess:
                    proc_name = "Process Exited"
            connections_data += f"Type: {conn.type}, Local: {conn.laddr}, Remote: {conn.raddr}, Status: {conn.status}, Process: {proc_name} (PID: {pid})\n"
        self.__save_data("network_connections_with_processes.txt", connections_data)
        log.info("Network connections with process info saved.")

    def __measure_network_bandwidth_usage(self, sample_count: int = 5, interval: float = 1.0):
        """
        Measures and saves the average network bandwidth usage.

        Args:
            sample_count: Number of samples to take (default: 5)
            interval: Time between samples in seconds (default: 1.0)
        """
        # TODO v3.4.1: Allow config.ini to set values
        log.debug("Measuring network bandwidth usage...")
        samples = []
        for _ in range(sample_count):
            net1 = psutil.net_io_counters()
            time.sleep(interval)
            net2 = psutil.net_io_counters()
            samples.append({
                'up': (net2.bytes_sent - net1.bytes_sent) / 1024,
                'down': (net2.bytes_recv - net1.bytes_recv) / 1024
            })
        if samples:
            avg_up = sum(s['up'] for s in samples) / len(samples)
            avg_down = sum(s['down'] for s in samples) / len(samples)
            max_up = max(s['up'] for s in samples)
            max_down = max(s['down'] for s in samples)
        else:
            avg_up = avg_down = max_up = max_down = 0
        bandwidth_data = f"Average Upload Speed: {avg_up:.2f} KB/s\n"
        bandwidth_data += f"Average Download Speed: {avg_down:.2f} KB/s\n"
        bandwidth_data += f"Peak Upload Speed: {max_up:.2f} KB/s\n"
        bandwidth_data += f"Peak Download Speed: {max_down:.2f} KB/s\n"
        self.__save_data("network_bandwidth_usage.txt", bandwidth_data)
        log.info("Network bandwidth usage saved.")

    def __fetch_hostname_and_ip(self):
        """
        Fetches and saves the hostname and IP addresses of the machine.
        """
        try:
            hostname = socket.gethostname()
            ip_addresses = []
            for res in socket.getaddrinfo(hostname, None):
                ip = res[4][0]
                if ip not in ip_addresses:
                    ip_addresses.append(ip)
            ip_config_data = f"Hostname: {hostname}\n"
            ip_config_data += "IP Addresses:\n"
            for ip in ip_addresses:
                ip_config_data += f"  - {ip}\n"
        except socket.gaierror as e:
            log.error(f"Failed to resolve hostname: {e}")
            ip_config_data = f"Hostname: {hostname}\nFailed to resolve IP addresses\n"
        self.__save_data("hostname_ip.txt", ip_config_data)
        log.info("Hostname and IP address saved.")


if __name__ == "__main__":
    NetworkInfo().get()
