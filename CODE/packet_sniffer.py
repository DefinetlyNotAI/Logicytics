from __future__ import annotations

import warnings
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from cryptography.utils import CryptographyDeprecationWarning

# TripleDES deprecation warning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scapy.all import sniff, conf
from scapy.layers.inet import IP, TCP, UDP, ICMP

from logicytics import log, config


class PacketSniffer:
    def __init__(self):
        conf.verb = 0
        self.packet_data = []
        self.G = nx.Graph()

    @staticmethod
    def _get_protocol(packet: IP) -> str:
        if packet.haslayer(TCP):
            return "TCP"
        elif packet.haslayer(UDP):
            return "UDP"
        elif packet.haslayer(ICMP):
            return "ICMP"
        return "Other"

    @staticmethod
    def _get_port(packet: IP, port_type: str) -> int | None:
        if port_type == "sport":
            return getattr(packet[TCP], "sport", None) if packet.haslayer(TCP) else getattr(packet[UDP], "sport", None)
        elif port_type == "dport":
            return getattr(packet[TCP], "dport", None) if packet.haslayer(TCP) else getattr(packet[UDP], "dport", None)
        return None

    def _log_packet(self, packet: IP):
        if not packet.haslayer(IP):
            return

        try:
            protocol = self._get_protocol(packet)
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

            src_port = dst_port = None
            if protocol in ("TCP", "UDP"):
                src_port = self._get_port(packet, "sport")
                dst_port = self._get_port(packet, "dport")

            info = {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "src_port": src_port,
                "dst_port": dst_port
            }

            self.packet_data.append(info)
            self.G.add_edge(src_ip, dst_ip, protocol=protocol)
            log.debug(f"{protocol} {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
        except Exception as err:
            log.error(f"Error logging packet: {err}")

    def _save_to_csv(self, path: str):
        if not self.packet_data:
            log.warning("No packets to save.")
            return
        pd.DataFrame(self.packet_data).to_csv(path, index=False)
        log.info(f"Saved packet data to {path}")

    def _visualize_graph(self, output: str = "graph.png"):
        if self.G.number_of_edges() == 0:
            log.warning("No edges to plot in graph.")
            return

        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, font_weight="bold")
        labels = nx.get_edge_attributes(self.G, 'protocol')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.title("Network Graph")
        plt.savefig(output)
        plt.close()
        log.info(f"Graph saved to {output}")

    @staticmethod
    def _correct_interface(iface: str) -> str:
        corrections = {"WiFi": "Wi-Fi", "Wi-Fi": "WiFi"}
        return corrections.get(iface, iface)

    def sniff_packets(self, iface: str, count: int, timeout: int, retry_max: int):
        iface = self._correct_interface(iface)
        retry_start = time()

        while time() - retry_start < retry_max:
            try:
                log.info(f"Sniffing on {iface}... (count={count}, timeout={timeout})")
                sniff(
                    iface=iface,
                    prn=self._log_packet,
                    count=count,
                    timeout=timeout
                )
                log.info("Sniff complete.")
                break
            except Exception as e:
                log.warning(f"Sniff failed on {iface}: {e}")
                iface = self._correct_interface(iface)
        else:
            log.error("Max retry time exceeded.")

        self._save_to_csv("packets.csv")
        self._visualize_graph()

    def run(self):
        iface = config.get("PacketSniffer Settings", "interface", fallback="WiFi")
        count = config.getint("PacketSniffer Settings", "packet_count", fallback=5000)
        timeout = config.getint("PacketSniffer Settings", "timeout", fallback=10)
        retry_max = config.getint("PacketSniffer Settings", "max_retry_time", fallback=30)

        if count <= 0 or timeout < 5 or retry_max < timeout:
            log.critical("Invalid configuration values.")
            return

        self.sniff_packets(iface, count, timeout, retry_max)

    def cleanup(self):
        self.G.clear()
        plt.close("all")
        log.info("Cleanup complete.")


if __name__ == "__main__":
    sniffer = PacketSniffer()
    try:
        sniffer.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        sniffer.cleanup()
