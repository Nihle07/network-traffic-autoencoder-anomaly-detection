from scapy.all import PcapReader, IP, TCP, UDP, DNS, DNSQR
import pandas as pd

PCAP_FILE = "/home/kali/mormal_session_2_yogun.pcap"
WINDOW_SIZE = 10  # saniye

rows = []

current_window_start = None

packet_count = 0
total_bytes = 0
tcp_count = 0
udp_count = 0
syn_count = 0
rst_count = 0
dns_query_count = 0

unique_domains = set()
unique_dst_ports = set()


def save_window(start_time):
    global packet_count, total_bytes, tcp_count, udp_count
    global syn_count, rst_count, dns_query_count
    global unique_domains, unique_dst_ports

    if packet_count == 0:
        return

    duration = WINDOW_SIZE

    rows.append({
        "window_start": start_time,
        "packet_count": packet_count,
        "bytes_per_second": total_bytes / duration,
        "average_packet_size": total_bytes / packet_count,
        "tcp_packet_count": tcp_count,
        "udp_packet_count": udp_count,
        "syn_count": syn_count,
        "rst_count": rst_count,
        "dns_query_count": dns_query_count,
        "unique_domains": len(unique_domains),
        "unique_destination_ports": len(unique_dst_ports)
    })

    packet_count = 0
    total_bytes = 0
    tcp_count = 0
    udp_count = 0
    syn_count = 0
    rst_count = 0
    dns_query_count = 0
    unique_domains = set()
    unique_dst_ports = set()


with PcapReader(PCAP_FILE) as pcap:
    for packet in pcap:

        timestamp = float(packet.time)

        if current_window_start is None:
            current_window_start = timestamp

        while timestamp >= current_window_start + WINDOW_SIZE:
            save_window(current_window_start)
            current_window_start += WINDOW_SIZE

        packet_count += 1
        total_bytes += len(packet)

        if TCP in packet:
            tcp_count += 1

            flags = packet[TCP].flags

            if flags & 0x02:  # SYN
                syn_count += 1

            if flags & 0x04:  # RST
                rst_count += 1

            if packet[TCP].dport:
                unique_dst_ports.add(packet[TCP].dport)

        elif UDP in packet:
            udp_count += 1

            if packet[UDP].dport:
                unique_dst_ports.add(packet[UDP].dport)

        if DNS in packet and packet[DNS].qr == 0:
            dns_query_count += 1

            if DNSQR in packet:
                try:
                    domain = packet[DNSQR].qname.decode(errors="ignore")
                    unique_domains.add(domain)
                except:
                    pass

save_window(current_window_start)

df = pd.DataFrame(rows)

df.to_csv("normal_session_2_yogun_features.csv", index=False)

print(df.head())
print()
print("Toplam pencere sayısı:", len(df))
print("normal_features.csv oluşturuldu")
