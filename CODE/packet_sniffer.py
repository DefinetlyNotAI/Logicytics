import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from adodbapi import InterfaceError
from scapy.all import sniff, conf
from scapy.layers.inet import IP, TCP, UDP, ICMP
from configparser import ConfigParser
from logicytics import *

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


# Read configuration from config.ini
config = ConfigParser()
config.read('config.ini')
config = config['PacketSniffer Settings']

# Global configuration
conf.verb = 0  # Turn off verbosity for clean output
packet_data = []  # List to store packet information
G = nx.Graph()  # Initialize a graph


# Function to process and log packet details
@log.function
def log_packet(packet: IP):
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
    except Exception as e:
        log.error(f"Error processing packet: {e}")


# Function to determine the protocol name
@log.function
def get_protocol_name(packet: IP):
    """Returns the name of the protocol."""
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
@log.function
def get_port_info(packet: IP, port_type: str):
    """Extracts the source or destination port from a packet."""
    log.debug(f"Port type: {port_type}")
    if packet.haslayer(TCP):
        return packet[TCP].sport if port_type == 'sport' else packet[TCP].dport
    elif packet.haslayer(UDP):
        return packet[UDP].sport if port_type == 'sport' else packet[UDP].dport
    return None


# Function to print packet summary
@log.function
def print_packet_summary(packet_info: dict):
    """Prints a summary of the captured packet."""
    log.info(f"Packet captured: {packet_info['protocol']} packet from {packet_info['src_ip']} "
             f"to {packet_info['dst_ip']} | Src Port: {packet_info['src_port']} | Dst Port: {packet_info['dst_port']}")


# Function to add packet information to the graph
@log.function
def add_to_graph(packet_info: dict):
    """Adds the packet information to the graph."""
    src_ip = packet_info['src_ip']
    dst_ip = packet_info['dst_ip']
    protocol = packet_info['protocol']
    G.add_edge(src_ip, dst_ip, protocol=protocol)


# Function to start sniffing packets
@log.function
def start_sniffing(interface: str, packet_count: int = 10, timeout: int = 10):
    """Starts packet sniffing on a given network interface."""
    log.info(f"Starting packet capture on interface '{interface}'...")

    # Initialize a packet capture counter
    packet_counter = 0

    # Define a custom packet callback to count packets
    def packet_callback(packet):
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
@log.function
def save_packet_data_to_csv(file_path: str):
    """Saves captured packet data to a CSV file."""
    global packet_data
    if packet_data:
        df = pd.DataFrame(packet_data)
        df.to_csv(file_path, index=False)
        log.info(f"Packet data saved to '{file_path}'.")
    else:
        log.warning("No packet data to save.")


# Function to visualize the graph
@log.function
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
    plt.show()


@log.function
def main():
    interface = config['interface']
    packet_count = int(config['packet_count'])
    timeout = int(config['timeout'])

    if packet_count == 0 or timeout == 0:
        log.error("Invalid packet count or timeout value. Please check the configuration.")

    try:
        start_sniffing(interface, packet_count, timeout)
    except InterfaceError as e:
        if interface != "WiFi" and interface != "Wi-Fi":
            log.error(f"Invalid interface '{interface}'. Please check the configuration: {e}")
        if interface == "WiFi":
            interface = "Wi-Fi"
        elif interface == "Wi-Fi":
            interface = "WiFi"
        start_sniffing(interface, packet_count, timeout)


# Entry point of the script
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(e)
