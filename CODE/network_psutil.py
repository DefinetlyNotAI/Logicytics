import os
import socket
import time

import psutil

from logicytics import Log, DEBUG, Execute

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


def __save_data(filename: str, data: str, father_dir_name: str = "network_data"):
    os.makedirs(father_dir_name, exist_ok=True)
    with open(os.path.join(father_dir_name, filename), "w") as f:
        f.write(data)


@log.function
def get_network_info():
    try:
        log.debug("Fetching network interface stats...")
        net_io = psutil.net_io_counters(pernic=True)
        net_io_data = ""
        for iface, stats in net_io.items():
            net_io_data += f"Interface: {iface}\n"
            net_io_data += f"Bytes Sent: {stats.bytes_sent}, Bytes Received: {stats.bytes_recv}\n"
            net_io_data += f"Packets Sent: {stats.packets_sent}, Packets Received: {stats.packets_recv}\n"
            net_io_data += f"Errors In: {stats.errin}, Errors Out: {stats.errout}\n"
            net_io_data += f"Dropped In: {stats.dropin}, Dropped Out: {stats.dropout}\n\n"
        __save_data("network_io.txt", net_io_data)
        log.info("Network IO stats saved.")

        log.debug("Fetching network connections...")
        connections = psutil.net_connections(kind='all')
        connections_data = ""
        for conn in connections:
            connections_data += f"Type: {conn.type}, Local: {conn.laddr}, Remote: {conn.raddr}, Status: {conn.status}\n"
        __save_data("network_connections.txt", connections_data)
        log.info("Network connections saved.")

        log.debug("Fetching network interface addresses...")
        interfaces = psutil.net_if_addrs()
        interfaces_data = ""
        for iface, addrs in interfaces.items():
            for addr in addrs:
                interfaces_data += f"Interface: {iface}, Address: {addr.address}, Netmask: {addr.netmask}, Broadcast: {addr.broadcast}\n"
        __save_data("network_interfaces.txt", interfaces_data)
        log.info("Network interface addresses saved.")

        log.debug("Fetching network interface stats...")
        stats = psutil.net_if_stats()
        stats_data = ""
        for iface, stat in stats.items():
            stats_data += f"Interface: {iface}, Speed: {stat.speed}Mbps, Duplex: {stat.duplex}, Up: {stat.isup}\n"
        __save_data("network_stats.txt", stats_data)
        log.info("Network interface stats saved.")

        log.debug("Executing external network command...")
        result = Execute.command("ipconfig")
        __save_data("network_command_output.txt", result)
        log.info("Network command output saved.")

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
        __save_data("network_connections_with_processes.txt", connections_data)
        log.info("Network connections with process info saved.")

        log.debug("Measuring network bandwidth usage...")
        net1 = psutil.net_io_counters()
        time.sleep(1)
        net2 = psutil.net_io_counters()
        bandwidth_data = f"Upload Speed: {(net2.bytes_sent - net1.bytes_sent) / 1024} KB/s\n"
        bandwidth_data += f"Download Speed: {(net2.bytes_recv - net1.bytes_recv) / 1024} KB/s\n"
        __save_data("network_bandwidth_usage.txt", bandwidth_data)
        log.info("Network bandwidth usage saved.")

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        ip_config_data = f"Hostname: {hostname}\nIP Address: {ip_address}\n"
        __save_data("hostname_ip.txt", ip_config_data)
        log.info("Hostname and IP address saved.")

    except Exception as e:
        log.error(f"Error getting network info: {e}")


if __name__ == "__main__":
    get_network_info()
