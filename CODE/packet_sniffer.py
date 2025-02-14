from __future__ import annotations

from configparser import ConfigParser

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scapy.all import sniff, conf
from scapy.layers.inet import IP, TCP, UDP, ICMP

from logicytics import log

# Read configuration from config.ini
config = ConfigParser()
config.read('config.ini')
config = config['PacketSniffer Settings']

# Global configuration
conf.verb = 0  # Turn off verbosity for clean output
packet_data = []  # List to store packet information
G = nx.Graph()  # Initialize a graph


# TODO Turn to a class
# Function to process and log packet details
def log_packet(packet: IP):
    """
    Processes a captured IP packet, extracting and logging network connection details.
    
    Extracts key network information from the packet including source and destination IP addresses, 
    protocol, source and destination ports. Logs packet details, updates global packet data collection, 
    prints a summary, and adds connection information to the network graph.
    
    Parameters:
        packet (IP): A Scapy IP layer packet to be processed and analyzed.
    
    Raises:
        Exception: Logs and suppresses any errors encountered during packet processing.
    
    Side Effects:
        - Appends packet information to global `packet_data` list
        - Prints packet summary to console
        - Updates network connection graph
        - Logs debug information about captured packet
    
    Notes:
        - Silently handles packet processing errors to prevent sniffing interruption
        - Requires global variables `packet_data` and supporting functions like 
          `get_protocol_name()`, `get_port_info()`, `print_packet_summary()`, and `add_to_graph()`
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
    Determines the protocol name of a captured network packet.
    
    This function examines the layers of a given IP packet to identify its protocol type. It supports identification of TCP, UDP, ICMP, and classifies any other packet types as 'Other'.
    
    Parameters:
        packet (IP): The captured network packet to analyze for protocol identification.
    
    Returns:
        str: The protocol name, which can be one of: 'TCP', 'UDP', 'ICMP', or 'Other'.
    
    Notes:
        - Uses Scapy's layer checking methods to determine protocol
        - Logs debug information about the packet and detected protocol
        - Provides a fallback 'Other' classification for unrecognized protocols
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
    
    Raises:
        ValueError: If an invalid port_type is provided.
    
    Notes:
        - Supports extracting ports from TCP and UDP layers
        - Returns None if the packet does not have TCP or UDP layers
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
    Prints a summary of the captured network packet to the debug log.
    
    Parameters:
        packet_info (dict): A dictionary containing detailed information about a captured network packet with the following expected keys:
            - 'protocol' (str): The network protocol of the packet (e.g., TCP, UDP, ICMP)
            - 'src_ip' (str): Source IP address of the packet
            - 'dst_ip' (str): Destination IP address of the packet
            - 'src_port' (int/str): Source port number of the packet
            - 'dst_port' (int/str): Destination port number of the packet
    
    Returns:
        None: Logs packet summary information without returning a value
    """
    log.debug(f"Packet captured: {packet_info['protocol']} packet from {packet_info['src_ip']} "
              f"to {packet_info['dst_ip']} | Src Port: {packet_info['src_port']} | Dst Port: {packet_info['dst_port']}")


# Function to add packet information to the graph
def add_to_graph(packet_info: dict):
    """
    Adds an edge to the network graph representing a connection between source and destination IPs.
    
    Parameters:
        packet_info (dict): A dictionary containing packet network details with the following keys:
            - 'src_ip' (str): Source IP address of the packet
            - 'dst_ip' (str): Destination IP address of the packet
            - 'protocol' (str): Network protocol used for the connection (e.g., TCP, UDP)
    
    Side Effects:
        Modifies the global NetworkX graph (G) by adding an edge between source and destination IPs
        with the protocol information as an edge attribute.
    
    Notes:
        - Assumes a global NetworkX graph object 'G' is already initialized
        - Does not perform validation of input packet_info dictionary
    """
    src_ip = packet_info['src_ip']
    dst_ip = packet_info['dst_ip']
    protocol = packet_info['protocol']
    G.add_edge(src_ip, dst_ip, protocol=protocol)


# Function to start sniffing packets
def start_sniffing(interface: str, packet_count: int = 10, timeout: int = 10):
    """
    Starts packet sniffing on a given network interface.
    
    Captures network packets on the specified interface with configurable packet count and timeout. Processes each captured packet using a custom callback function, logs packet details, and stops when the specified packet count is reached.
    
    Parameters:
        interface (str): Network interface name to capture packets from.
        packet_count (int, optional): Maximum number of packets to capture. Defaults to 10.
        timeout (int, optional): Maximum time to spend capturing packets in seconds. Defaults to 10.
    
    Side Effects:
        - Logs packet details during capture
        - Saves captured packet data to a CSV file
        - Generates a network graph visualization
    
    Raises:
        Exception: If packet capture encounters unexpected errors
    
    Example:
        start_sniffing('eth0', packet_count=50, timeout=30)
    """
    log.info(f"Starting packet capture on interface '{interface}'...")

    # Initialize a packet capture counter
    packet_counter = 0

    # Define a custom packet callback to count packets
    def packet_callback(packet: IP) -> bool:
        """
        Callback function to process each captured network packet during sniffing.
        
        Processes individual packets, logs their details, and manages packet capture termination. Tracks the number of packets captured and stops sniffing when the predefined packet count is reached.
        
        Parameters:
            packet (IP): The captured network packet to be processed.
        
        Returns:
            bool: True if the specified packet count has been reached, signaling the sniffer to stop; False otherwise.
        
        Side Effects:
            - Increments the global packet counter
            - Logs packet details using log_packet function
            - Logs debug information about received packets
            - Stops packet capture when packet count limit is met
        
        Raises:
            No explicit exceptions raised, but may propagate exceptions from log_packet function.
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
    
    Writes the collected network packet information to a specified CSV file. If packet data exists, it creates a pandas DataFrame and exports it to the given file path. If no packet data has been captured, it logs a warning message.
    
    Parameters:
        file_path (str): The file path where the packet data will be saved as a CSV file.
    
    Returns:
        None
    
    Side Effects:
        - Writes packet data to a CSV file
        - Logs an informational message on successful save
        - Logs a warning if no packet data is available
    
    Raises:
        IOError: Potential file writing permission or path-related errors (implicitly handled by pandas)
    """
    global packet_data
    if packet_data:
        df = pd.DataFrame(packet_data)
        df.to_csv(file_path, index=False)
        log.info(f"Packet data saved to '{file_path}'.")
    else:
        log.warning("No packet data to save.")


# Function to visualize the graph of packet connections
def visualize_graph(node_colors: dict[str, str] | None = None,
                    node_sizes: dict[str, int] | None = None,
                    *,  # Force keyword arguments for the following parameters
                    figsize: tuple[int, int] = (12, 8),
                    font_size: int = 10,
                    font_weight: str = "bold",
                    title: str = "Network Connections Graph",
                    output_file: str = "network_connections_graph.png",
                    layout_func: callable = nx.spring_layout):
    """
    Visualizes the graph of packet connections with customizable node colors and sizes.
    
    Generates a network graph representation of packet connections using NetworkX and Matplotlib, with optional customization of node colors and sizes.
    
    Parameters:
        node_colors (dict, optional): A dictionary mapping nodes to their display colors. 
            If not provided, defaults to skyblue for all nodes.
        node_sizes (dict, optional): A dictionary mapping nodes to their display sizes. 
            If not provided, defaults to 3000 for all nodes.
        figsize (tuple, optional): The size of the figure in inches (width, height). Defaults to (12, 8).
        font_size (int, optional): The font size for node labels. Defaults to 10.
        font_weight (str, optional): The font weight for node labels. Defaults to 'bold'.
        title (str, optional): The title of the graph. Defaults to 'Network Connections Graph'.
        output_file (str, optional): The name of the output PNG file to save the graph visualization. Defaults to 'network_connections_graph.png'.
        layout_func (callable, optional): The layout function to use for the graph. Defaults to nx.spring_layout.

    Side Effects:
        - Creates a matplotlib figure
        - Saves a PNG image file named 'network_connections_graph.png'
        - Closes the matplotlib figure after saving
    
    Returns:
        None
    
    Example:
        # Default visualization
        visualize_graph()
    
        # Custom node colors and sizes
        custom_colors = {'192.168.1.1': 'red', '10.0.0.1': 'green'}
        custom_sizes = {'192.168.1.1': 5000, '10.0.0.1': 2000}
        visualize_graph(node_colors=custom_colors, node_sizes=custom_sizes)
    """
    pos = layout_func(G)
    plt.figure(figsize=figsize)

    if node_colors is None:
        node_colors = {node: "skyblue" for node in G.nodes()}

    if node_sizes is None:
        node_sizes = {node: 3000 for node in G.nodes()}

    node_color_list = [node_colors.get(node, "skyblue") for node in G.nodes()]
    node_size_list = [node_sizes.get(node, 3000) for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_size=node_size_list, node_color=node_color_list, font_size=font_size,
            font_weight=font_weight)
    edge_labels = nx.get_edge_attributes(G, 'protocol')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.savefig(output_file)
    plt.close()


@log.function
def packet_sniffer():
    """
    Initiates packet sniffing based on configuration settings.
    
    Reads network configuration parameters from a global config dictionary, including network interface, packet count, and timeout. Validates input parameters to ensure they are positive values. Attempts to start packet sniffing on the specified interface, with built-in error handling and interface name correction for common variations.

    Raises:
        SystemExit: If packet count or timeout values are invalid
        Exception: If there are issues with the network interface or packet sniffing process
    
    Side Effects:
        - Logs configuration and sniffing errors
        - Attempts to autocorrect interface names
        - Calls start_sniffing() to capture network packets
        - Exits the program if critical configuration errors are encountered
    """

    def correct_interface_name(interface_name: str) -> str:
        corrections = {
            "WiFi": "Wi-Fi",
            "Wi-Fi": "WiFi"
        }
        return corrections.get(interface_name, interface_name)

    interface = config['interface']
    packet_count = int(config['packet_count'])
    timeout = int(config['timeout'])

    if packet_count <= 0 or timeout <= 0:
        try:
            log.error(
                "Oops! Can't work with these values (Not your fault):\n"
                f"          - Packet count: {packet_count} {'❌ (must be > 0)' if packet_count <= 0 else '✅'}\n"
                f"          - Timeout: {timeout} {'❌ (must be > 0)' if timeout <= 0 else '✅'}"
            )
        except Exception:
            log.error("Error reading configuration: Improper values for packet count or timeout")
        exit(1)

    for attempt in range(2):  # Try original and corrected name
        try:
            start_sniffing(interface, packet_count, timeout)
            break
        except Exception as err:
            if attempt == 0 and interface in ("WiFi", "Wi-Fi"):
                log.warning("Retrying with corrected interface name...")
                interface = correct_interface_name(interface)
            else:
                log.error(f"Failed to sniff packets: {err}")


# Entry point of the script
if __name__ == "__main__":
    try:
        packet_sniffer()
    except Exception as e:
        log.error(e)
    finally:
        # Clean up resources
        try:
            if G:
                plt.close('all')  # Close all figures
        except Exception as e:
            log.error(f"Error during cleanup: {e}")
