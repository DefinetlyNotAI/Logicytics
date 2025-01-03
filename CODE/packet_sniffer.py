from __future__ import annotations

from configparser import ConfigParser

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scapy.all import sniff, conf
from scapy.layers.inet import IP, TCP, UDP, ICMP

from logicytics import Log, DEBUG

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})

try:
    # Read configuration from config.ini
    config = ConfigParser()
    config.read('config.ini')
    config = config['PacketSniffer Settings']

    # Global configuration
    conf.verb = 0  # Turn off verbosity for clean output
    packet_data = []  # List to store packet information
    G = nx.Graph()  # Initialize a graph
except Exception as e:
    log.error(f"Error reading configuration: {e}")
    exit(1)


# Function to process and log packet details
def log_packet(packet: IP):
    """
    Processes a captured packet, logs its details, and updates the packet data and graph.

    Parameters:
    packet (IP): The captured packet to process.

    Returns:
    None
    """
    try:
        if packet.haslayer(IP):
            log.debug(f"Packet captured: {packet.summary()}")
            packet_info = {
                'src_ip': packet[IP].src,
                'dst_ip': packet[IP].dst,
                'protocol': get_protocol_name(packet),
                'src_port': get_port_info(packet, 'sport'),
                'dst_port': get_port_info(packet, 'dport'),
            }
            packet_data.append(packet_info)
            print_packet_summary(packet_info)
            add_to_graph(packet_info)
    except Exception as err:
        log.error(f"Error processing packet: {err}")


# Function to determine the protocol name
def get_protocol_name(packet: IP) -> str:
    """
    Determines the protocol name of a captured packet.

    Parameters:
    packet (IP): The captured packet to analyze.

    Returns:
    str: The name of the protocol (TCP, UDP, ICMP, or Other).
    """
    log.debug(f"Checking protocol for packet: {packet.summary()}")
    if packet.haslayer(TCP):
        log.debug("Protocol: TCP")
        return 'TCP'
    elif packet.haslayer(UDP):
        log.debug("Protocol: UDP")
        return 'UDP'
    elif packet.haslayer(ICMP):
        log.debug("Protocol: ICMP")
        return 'ICMP'
    else:
        log.debug("Protocol: Other")
        return 'Other'


# Function to extract port information from a packet
def get_port_info(packet: IP, port_type: str) -> int | None:
    """
    Extracts the source or destination port from a captured packet.

    Parameters:
    packet (IP): The captured packet to analyze.
    port_type (str): The type of port to extract ('sport' for source port, 'dport' for destination port).

    Returns:
    int | None: The port number if available, otherwise None.
    """
    log.debug(f"Port type: {port_type}")
    if packet.haslayer(TCP):
        return packet[TCP].sport if port_type == 'sport' else packet[TCP].dport
    elif packet.haslayer(UDP):
        return packet[UDP].sport if port_type == 'sport' else packet[UDP].dport
    return None


# Function to print packet summary
def print_packet_summary(packet_info: dict):
    """
    Prints a summary of the captured packet.

    Parameters:
    packet_info (dict): A dictionary containing packet details.

    Returns:
    None
    """
    log.debug(f"Packet captured: {packet_info['protocol']} packet from {packet_info['src_ip']} "
              f"to {packet_info['dst_ip']} | Src Port: {packet_info['src_port']} | Dst Port: {packet_info['dst_port']}")


# Function to add packet information to the graph
def add_to_graph(packet_info: dict):
    """
    Adds the packet information to the graph.

    Parameters:
    packet_info (dict): A dictionary containing packet details.

    Returns:
    None
    """
    src_ip = packet_info['src_ip']
    dst_ip = packet_info['dst_ip']
    protocol = packet_info['protocol']
    G.add_edge(src_ip, dst_ip, protocol=protocol)


# Function to start sniffing packets
def start_sniffing(interface: str, packet_count: int = 10, timeout: int = 10):
    """
    Starts packet sniffing on a given network interface.

    Parameters:
    interface (str): The network interface to sniff on.
    packet_count (int): The number of packets to capture.
    timeout (int): The timeout for packet capture in seconds.

    Returns:
    None
    """
    log.info(f"Starting packet capture on interface '{interface}'...")

    # Initialize a packet capture counter
    packet_counter = 0

    # Define a custom packet callback to count packets
    def packet_callback(packet: IP) -> bool:
        """
        Callback function to process each captured packet.

        Parameters:
        packet (IP): The captured packet.

        Returns:
        bool: True if the packet count is reached, otherwise False.
        """
        log.debug(f"Received packet: {packet.summary()}")
        nonlocal packet_counter  # Reference the outer packet_counter
        if packet_counter >= packet_count:
            # Stop sniffing once the packet count is reached
            log.info(f"Captured {packet_count} packets, stopping sniffing.")
            return True  # Return True to stop sniffing
        log_packet(packet)  # Call the existing log_packet function
        packet_counter += 1  # Increment the packet counter

    # Start sniffing with the custom callback
    sniff(iface=interface, prn=packet_callback, count=packet_count, timeout=timeout)

    # After sniffing completes, save the captured packet data to CSV and visualize the graph
    log.info("Packet capture completed.")
    save_packet_data_to_csv('captured_packets.csv')
    visualize_graph()


# Function to save captured packet data to CSV
def save_packet_data_to_csv(file_path: str):
    """
    Saves captured packet data to a CSV file.

    Parameters:
    file_path (str): The path to the CSV file where the packet data will be saved.

    Returns:
    None
    """
    global packet_data
    if packet_data:
        df = pd.DataFrame(packet_data)
        df.to_csv(file_path, index=False)
        log.info(f"Packet data saved to '{file_path}'.")
    else:
        log.warning("No packet data to save.")


# Function to visualize the graph
def visualize_graph(node_colors: str = None, node_sizes: str = None):
    """
    Visualizes the graph of packet connections with customizable node colors and sizes.

    Parameters:
    node_colors (dict): A dictionary mapping node to color.
    node_sizes (dict): A dictionary mapping node to size.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    if node_colors is None:
        node_colors = {node: "skyblue" for node in G.nodes()}

    if node_sizes is None:
        node_sizes = {node: 3000 for node in G.nodes()}

    node_color_list = [node_colors.get(node, "skyblue") for node in G.nodes()]
    node_size_list = [node_sizes.get(node, 3000) for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_size=node_size_list, node_color=node_color_list, font_size=10,
            font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'protocol')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Network Connections Graph")
    plt.savefig("network_connections_graph.png")
    plt.close()


@log.function
def packet_sniffer():
    """
    Main function to read configuration and start packet sniffing.

    Reads the network interface, packet count, and timeout from the configuration.
    Validates the packet count and timeout values.
    Starts packet sniffing on the specified interface.
    Handles exceptions related to invalid interface names and attempts to correct them.
    """
    interface = config['interface']
    packet_count = int(config['packet_count'])
    timeout = int(config['timeout'])

    if packet_count <= 0 or timeout <= 0:
        try:
            log.error(
                "Oops! Can't work with these values:\n"
                f"- Packet count: {packet_count} {'❌ (must be > 0)' if packet_count <= 0 else '✅'}\n"
                f"- Timeout: {timeout} {'❌ (must be > 0)' if timeout <= 0 else '✅'}"
            )
        except Exception:
            log.error("Error reading configuration: Improper values for packet count or timeout")
        exit(1)

    try:
        start_sniffing(interface, packet_count, timeout)
    except Exception as err:
        log.error(f"Invalid interface '{interface}'. Please check the configuration: {err}")
        if interface == "WiFi" or interface == "Wi-Fi":
            log.warning("Attempting to correct the interface name...")
            interface = "Wi-Fi" if interface == "WiFi" else "WiFi"
            log.info(f"Interface name auto-corrected to '{interface}', retrying packet sniffing...")
            try:
                start_sniffing(interface, packet_count, timeout)
            except Exception as err:
                log.error(f"Error sniffing packets on auto-corrected interface '{interface}': {err}")


# Entry point of the script
if __name__ == "__main__":
    try:
        packet_sniffer()
    except Exception as e:
        log.error(e)
        exit(1)
