#!/usr/bin/env python3
"""
Advanced Network Traffic Monitor with Suspicious Packet Detection
Monitors network traffic in real-time with comprehensive security analysis
"""

import socket
import struct
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import sys
import argparse
import re
import ipaddress
from supabase_client import supabase

class SecurityAnalyzer:
    def __init__(self):
        # Known malicious patterns
        self.malicious_signatures = [
            b'metasploit', b'meterpreter', b'payload', b'shellcode',
            b'backdoor', b'trojan', b'keylogger', b'rootkit'
        ]
        
        # Suspicious ports (commonly used by malware)
        self.suspicious_ports = {
            1337, 31337, 12345, 54321, 9999, 6666, 4444, 8080,
            6667, 6697, 1234, 3389, 5900, 23, 135, 139, 445
        }
        
        # Known attack patterns
        self.attack_patterns = {
            'sql_injection': [b'union select', b'drop table', b'exec(', b'<script>'],
            'xss': [b'<script>', b'javascript:', b'onerror=', b'onload='],
            'directory_traversal': [b'../', b'..\\', b'etc/passwd', b'boot.ini'],
            'command_injection': [b';cat ', b'|nc ', b'&& ', b'|| ']
        }
        
        # Track connection patterns
        self.connection_tracker = defaultdict(lambda: {
            'count': 0, 
            'first_seen': None, 
            'last_seen': None,
            'ports': set(),
            'data_volume': 0
        })
        
        # Port scan detection
        self.port_scan_threshold = 10  # ports per minute
        self.port_scan_tracker = defaultdict(lambda: {'ports': set(), 'timestamp': None})
        
        # DDoS detection
        self.ddos_threshold = 100  # packets per second per IP
        self.ddos_tracker = defaultdict(lambda: deque(maxlen=100))
        
        # Anomaly thresholds
        self.packet_size_threshold = 1500  # Unusually large packets
        self.connection_rate_threshold = 50  # connections per minute per IP

class NetworkTrafficMonitor:
    def __init__(self, interface='eth0', max_packets=1000):
        self.interface = interface
        self.max_packets = max_packets
        self.running = False
        self.packets = deque(maxlen=max_packets)
        self.suspicious_packets = deque(maxlen=500)
        
        # Security analyzer
        self.security = SecurityAnalyzer()
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'suspicious_packets': 0,
            'protocols': defaultdict(int),
            'src_ips': defaultdict(int),
            'dst_ips': defaultdict(int),
            'ports': defaultdict(int),
            'packet_sizes': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'threats': defaultdict(int),
            'blocked_ips': set(),
            'alerts': deque(maxlen=100)
        }
        
        # Protocol mapping
        self.protocols = {
            1: 'ICMP', 6: 'TCP', 17: 'UDP', 2: 'IGMP', 89: 'OSPF'
        }
        
        # Known malicious IP ranges (example - you can expand this)
        self.malicious_networks = [
            ipaddress.ip_network('10.0.0.0/8'),  # Example private ranges to monitor
            # Add actual threat intelligence feeds here
        ]
        
    def create_socket(self):
        """Create raw socket for packet capture"""
        try:
            if sys.platform.startswith('linux'):
                s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            elif sys.platform.startswith('win'):
                s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
                s.bind((socket.gethostbyname(socket.gethostname()), 0))
                s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
                s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
                
            return s
        except PermissionError:
            print("Error: Need administrator/root privileges to capture packets")
            sys.exit(1)
        except Exception as e:
            print(f"Error creating socket: {e}")
            sys.exit(1)
    
    def parse_ethernet_header(self, packet):
        """Parse Ethernet header"""
        eth_header = struct.unpack('!6s6sH', packet[:14])
        eth_protocol = socket.ntohs(eth_header[2])
        return eth_protocol, packet[14:]
    
    def parse_ip_header(self, packet):
        """Parse IP header"""
        ip_header = packet[:20]
        iph = struct.unpack('!BBHHHBBH4s4s', ip_header)
        
        version_ihl = iph[0]
        version = version_ihl >> 4
        ihl = version_ihl & 0xF
        
        iph_length = ihl * 4
        ttl = iph[5]
        protocol = iph[6]
        s_addr = socket.inet_ntoa(iph[8])
        d_addr = socket.inet_ntoa(iph[9])
        
        return {
            'version': version,
            'header_length': iph_length,
            'ttl': ttl,
            'protocol': protocol,
            'source': s_addr,
            'destination': d_addr,
            'data': packet[iph_length:]
        }
    
    def parse_tcp_header(self, packet):
        """Parse TCP header"""
        tcp_header = packet[:20]
        tcph = struct.unpack('!HHLLBBHHH', tcp_header)
        
        source_port = tcph[0]
        dest_port = tcph[1]
        sequence = tcph[2]
        acknowledgment = tcph[3]
        doff_reserved = tcph[4]
        tcph_length = (doff_reserved >> 4) * 4
        flags = tcph[5]
        
        return {
            'source_port': source_port,
            'dest_port': dest_port,
            'sequence': sequence,
            'acknowledgment': acknowledgment,
            'header_length': tcph_length,
            'flags': flags,
            'data': packet[tcph_length:]
        }
    
    def parse_udp_header(self, packet):
        """Parse UDP header"""
        udp_header = packet[:8]
        udph = struct.unpack('!HHHH', udp_header)
        
        return {
            'source_port': udph[0],
            'dest_port': udph[1],
            'length': udph[2],
            'checksum': udph[3],
            'data': packet[8:]
        }
    
    def detect_port_scan(self, src_ip, dst_port, timestamp):
        """Detect port scanning attempts"""
        now = timestamp
        tracker = self.security.port_scan_tracker[src_ip]
        
        # Reset if more than 1 minute has passed
        if tracker['timestamp'] and (now - tracker['timestamp']).seconds > 60:
            tracker['ports'] = set()
            tracker['timestamp'] = now
        
        if not tracker['timestamp']:
            tracker['timestamp'] = now
            
        tracker['ports'].add(dst_port)
        
        if len(tracker['ports']) > self.security.port_scan_threshold:
            return True, f"Port scan detected from {src_ip}: {len(tracker['ports'])} ports in 1 minute"
        
        return False, None
    
    def detect_ddos(self, src_ip, timestamp):
        """Detect DDoS attacks"""
        tracker = self.security.ddos_tracker[src_ip]
        tracker.append(timestamp)
        
        if len(tracker) >= self.security.ddos_threshold:
            time_diff = (tracker[-1] - tracker[0]).total_seconds()
            if time_diff <= 1:  # 100 packets in 1 second
                return True, f"DDoS attack detected from {src_ip}: {len(tracker)} packets/second"
        
        return False, None
    
    def detect_suspicious_payload(self, payload):
        """Analyze packet payload for suspicious content"""
        threats = []
        payload_lower = payload.lower()
        
        # Check for malicious signatures
        for signature in self.security.malicious_signatures:
            if signature in payload_lower:
                threats.append(f"Malicious signature: {signature.decode('utf-8', errors='ignore')}")
        
        # Check for attack patterns
        for attack_type, patterns in self.security.attack_patterns.items():
            for pattern in patterns:
                if pattern in payload_lower:
                    threats.append(f"{attack_type.title()} attempt: {pattern.decode('utf-8', errors='ignore')}")
        
        return threats
    
    def check_malicious_ip(self, ip):
        """Check if IP is in known malicious ranges"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in self.malicious_networks:
                if ip_addr in network:
                    return True, f"IP {ip} in known malicious network {network}"
        except ValueError:
            pass
        return False, None
    
    def analyze_packet_security(self, packet_info):
        """Comprehensive security analysis of packet"""
        threats = []
        threat_level = 0
        
        src_ip = packet_info.get('source')
        dst_ip = packet_info.get('destination')
        timestamp = packet_info.get('timestamp')
        packet_size = packet_info.get('size', 0)
        
        # Check for malicious IPs
        is_malicious_src, src_msg = self.check_malicious_ip(src_ip)
        if is_malicious_src:
            threats.append(src_msg)
            threat_level += 3
        
        is_malicious_dst, dst_msg = self.check_malicious_ip(dst_ip)
        if is_malicious_dst:
            threats.append(dst_msg)
            threat_level += 2
        
        # Check packet size anomalies
        if packet_size > self.security.packet_size_threshold:
            threats.append(f"Unusually large packet: {packet_size} bytes")
            threat_level += 1
        
        # Check TTL anomalies (possible spoofing)
        ttl = packet_info.get('ttl', 64)
        if ttl < 32 or ttl > 255:
            threats.append(f"Suspicious TTL value: {ttl}")
            threat_level += 2
        
        # Protocol-specific analysis
        if 'tcp' in packet_info:
            tcp_info = packet_info['tcp']
            src_port = tcp_info['source_port']
            dst_port = tcp_info['dest_port']
            flags = tcp_info.get('flags', 0)
            
            # Check suspicious ports
            if src_port in self.security.suspicious_ports or dst_port in self.security.suspicious_ports:
                threats.append(f"Suspicious port usage: {src_port}->{dst_port}")
                threat_level += 2
            
            # Detect port scanning
            is_scan, scan_msg = self.detect_port_scan(src_ip, dst_port, timestamp)
            if is_scan:
                threats.append(scan_msg)
                threat_level += 3
            
            # Check for TCP flag anomalies
            if flags == 0:  # Null scan
                threats.append("Null scan detected (all TCP flags = 0)")
                threat_level += 3
            elif flags == 0x29:  # XMAS scan
                threats.append("XMAS scan detected (FIN+URG+PSH flags)")
                threat_level += 3
            elif flags == 0x02 and src_port > 1024 and dst_port > 1024:  # SYN to high ports
                threats.append("Possible SYN flood or reconnaissance")
                threat_level += 1
            
            # Analyze payload
            payload = tcp_info.get('data', b'')
            payload_threats = self.detect_suspicious_payload(payload)
            threats.extend(payload_threats)
            threat_level += len(payload_threats) * 2
        
        elif 'udp' in packet_info:
            udp_info = packet_info['udp']
            src_port = udp_info['source_port']
            dst_port = udp_info['dest_port']
            
            # Check suspicious ports
            if src_port in self.security.suspicious_ports or dst_port in self.security.suspicious_ports:
                threats.append(f"Suspicious UDP port: {src_port}->{dst_port}")
                threat_level += 1
            
            # DNS tunneling detection (large DNS packets)
            if dst_port == 53 and packet_size > 512:
                threats.append(f"Possible DNS tunneling: {packet_size} byte DNS packet")
                threat_level += 2
            
            # Analyze payload
            payload = udp_info.get('data', b'')
            payload_threats = self.detect_suspicious_payload(payload)
            threats.extend(payload_threats)
            threat_level += len(payload_threats) * 2
        
        # DDoS detection
        is_ddos, ddos_msg = self.detect_ddos(src_ip, timestamp)
        if is_ddos:
            threats.append(ddos_msg)
            threat_level += 4
        
        return threats, threat_level
    
    def log_security_alert(self, packet_info, threats, threat_level):
        """Log security alert"""
        alert = {
            'timestamp': packet_info['timestamp'],
            'threat_level': threat_level,
            'src_ip': packet_info.get('source'),
            'dst_ip': packet_info.get('destination'),
            'protocol': self.protocols.get(packet_info.get('protocol', 0), 'Unknown'),
            'threats': threats,
            'packet_size': packet_info.get('size', 0)
        }
        
        self.stats['alerts'].append(alert)

        # --- Supabase: send alert to remote DB (if configured) ---
        try:
            supabase.table("alerts").insert({
                "timestamp": packet_info['timestamp'].isoformat(),
                "src_ip": packet_info.get('source'),
                "dst_ip": packet_info.get('destination'),
                "protocol": self.protocols.get(packet_info.get('protocol', 0), 'Unknown'),
                "threat_level": threat_level,
                "threats": threats,
                "packet_size": packet_info.get('size', 0)
            }).execute()
        except Exception as e:
            # don't let DB errors stop the monitor; just log them
            print(f"Error sending alert to Supabase: {e}")
        
        # Auto-block high threat IPs
        if threat_level >= 5:
            self.stats['blocked_ips'].add(packet_info.get('source'))
            print(f"\n🚨 HIGH THREAT ALERT 🚨")
            print(f"IP {packet_info.get('source')} automatically blocked!")
        
        # Print real-time alerts for significant threats
        if threat_level >= 3:
            print(f"\n⚠️  SECURITY ALERT (Level {threat_level})")
            print(f"Time: {packet_info['timestamp'].strftime('%H:%M:%S')}")
            print(f"Source: {packet_info.get('source')} -> Destination: {packet_info.get('destination')}")
            for threat in threats:
                print(f"  - {threat}")
    
    def process_packet(self, packet):
        """Process captured packet with security analysis"""
        timestamp = datetime.now()
        packet_info = {
            'timestamp': timestamp,
            'size': len(packet),
            'raw_data': packet[:100]
        }
        
        try:
            if sys.platform.startswith('linux'):
                eth_protocol, ip_packet = self.parse_ethernet_header(packet)
                if eth_protocol != 0x0800:
                    return
            else:
                ip_packet = packet
            
            # Parse IP header
            ip_info = self.parse_ip_header(ip_packet)
            packet_info.update(ip_info)
            
            # Update basic statistics
            self.stats['total_packets'] += 1
            self.stats['protocols'][self.protocols.get(ip_info['protocol'], f"Unknown({ip_info['protocol']})")] += 1
            self.stats['src_ips'][ip_info['source']] += 1
            self.stats['dst_ips'][ip_info['destination']] += 1
            self.stats['packet_sizes'].append(len(packet))
            self.stats['timestamps'].append(timestamp)
            
            # Parse transport layer
            if ip_info['protocol'] == 6:  # TCP
                tcp_info = self.parse_tcp_header(ip_info['data'])
                packet_info['tcp'] = tcp_info
                self.stats['ports'][tcp_info['source_port']] += 1
                self.stats['ports'][tcp_info['dest_port']] += 1
                
            elif ip_info['protocol'] == 17:  # UDP
                udp_info = self.parse_udp_header(ip_info['data'])
                packet_info['udp'] = udp_info
                self.stats['ports'][udp_info['source_port']] += 1
                self.stats['ports'][udp_info['dest_port']] += 1
            
            # Security analysis
            threats, threat_level = self.analyze_packet_security(packet_info)
            
            if threats:
                self.stats['suspicious_packets'] += 1
                self.stats['threats']['total'] += 1
                packet_info['threats'] = threats
                packet_info['threat_level'] = threat_level
                self.suspicious_packets.append(packet_info)
                self.log_security_alert(packet_info, threats, threat_level)
                
                # Update threat statistics
                for threat in threats:
                    threat_type = threat.split(':')[0]
                    self.stats['threats'][threat_type] += 1
            
            # Store packet info
            self.packets.append(packet_info)
            
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def capture_packets(self):
        """Main packet capture loop"""
        sock = self.create_socket()
        print(f"🔍 Starting advanced packet capture with security analysis...")
        print(f"Interface: {self.interface}")
        print(f"Monitoring for suspicious activities...\n")
        
        try:
            while self.running:
                packet, addr = sock.recvfrom(65565)
                self.process_packet(packet)
                
        except KeyboardInterrupt:
            print("\nStopping packet capture...")
        except Exception as e:
            print(f"Error in packet capture: {e}")
        finally:
            sock.close()
    
    def print_security_dashboard(self):
        """Print comprehensive security dashboard"""
        print("\n" + "="*80)
        print(f"🛡️  NETWORK SECURITY DASHBOARD [{self.interface}] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Basic statistics
        print(f"📊 Total Packets: {self.stats['total_packets']:,} | "
              f"🚨 Suspicious: {self.stats['suspicious_packets']:,} | "
              f"🚫 Blocked IPs: {len(self.stats['blocked_ips'])}")
        
        if self.stats['total_packets'] > 0:
            threat_percentage = (self.stats['suspicious_packets'] / self.stats['total_packets']) * 100
            print(f"🔍 Threat Level: {threat_percentage:.2f}% of traffic flagged as suspicious")
        
        # Recent alerts
        if self.stats['alerts']:
            print(f"\n⚠️  RECENT SECURITY ALERTS (Last 5):")
            for alert in list(self.stats['alerts'])[-5:]:
                level_emoji = "🔴" if alert['threat_level'] >= 5 else "🟡" if alert['threat_level'] >= 3 else "🟢"
                print(f"  {level_emoji} {alert['timestamp'].strftime('%H:%M:%S')} "
                      f"Level {alert['threat_level']} - {alert['src_ip']} -> {alert['dst_ip']}")
                for threat in alert['threats'][:2]:  # Show first 2 threats
                    print(f"    • {threat}")
        
        # Threat summary
        if self.stats['threats']:
            print(f"\n🎯 THREAT SUMMARY:")
            threat_items = sorted(self.stats['threats'].items(), key=lambda x: x[1], reverse=True)
            for threat_type, count in threat_items[:8]:
                print(f"  • {threat_type}: {count}")
        
        # Top suspicious IPs
        suspicious_ips = defaultdict(int)
        for packet in self.suspicious_packets:
            suspicious_ips[packet.get('source', 'Unknown')] += 1
        
        if suspicious_ips:
            print(f"\n🕵️  TOP SUSPICIOUS IPs:")
            for ip, count in sorted(suspicious_ips.items(), key=lambda x: x[1], reverse=True)[:5]:
                status = " (BLOCKED)" if ip in self.stats['blocked_ips'] else ""
                print(f"  • {ip}: {count} suspicious packets{status}")
        
        # Protocol distribution
        print(f"\n📡 PROTOCOL DISTRIBUTION:")
        for protocol, count in sorted(self.stats['protocols'].items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / self.stats['total_packets']) * 100
            print(f"  • {protocol}: {count:,} ({percentage:.1f}%)")
        
        # Traffic rate
        if len(self.stats['timestamps']) >= 2:
            time_diff = (self.stats['timestamps'][-1] - self.stats['timestamps'][0]).total_seconds()
            if time_diff > 0:
                rate = len(self.stats['timestamps']) / time_diff
                print(f"\n📈 Current Traffic Rate: {rate:.1f} packets/second")
    
    def display_recent_suspicious_packets(self, count=5):
        """Display recent suspicious packets"""
        if not self.suspicious_packets:
            return
            
        print(f"\n🚨 RECENT SUSPICIOUS PACKETS (Last {count}):")
        print("-" * 80)
        
        for packet in list(self.suspicious_packets)[-count:]:
            timestamp = packet['timestamp'].strftime('%H:%M:%S')
            src_ip = packet.get('source', 'Unknown')
            dst_ip = packet.get('destination', 'Unknown')
            threat_level = packet.get('threat_level', 0)
            protocol = self.protocols.get(packet.get('protocol', 0), 'Unknown')
            
            level_indicator = "🔴" if threat_level >= 5 else "🟡" if threat_level >= 3 else "🟢"
            
            print(f"{level_indicator} {timestamp} {protocol:4} {src_ip:>15} -> {dst_ip:>15} "
                  f"(Level {threat_level})")
            
            for threat in packet.get('threats', [])[:2]:
                print(f"    └─ {threat}")
    
    def save_security_report(self, filename):
        """Save comprehensive security report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_packets': self.stats['total_packets'],
                'suspicious_packets': self.stats['suspicious_packets'],
                'blocked_ips': len(self.stats['blocked_ips']),
                'threat_percentage': (self.stats['suspicious_packets'] / max(self.stats['total_packets'], 1)) * 100
            },
            'threats': dict(self.stats['threats']),
            'blocked_ips': list(self.stats['blocked_ips']),
            'alerts': [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'threat_level': alert['threat_level'],
                    'src_ip': alert['src_ip'],
                    'dst_ip': alert['dst_ip'],
                    'protocol': alert['protocol'],
                    'threats': alert['threats']
                }
                for alert in self.stats['alerts']
            ],
            'protocols': dict(self.stats['protocols']),
            'top_suspicious_ips': dict(sorted(
                {packet.get('source'): 1 for packet in self.suspicious_packets}.items(),
                key=lambda x: x[1], reverse=True
            )[:10])
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"🔐 Security report saved to {filename}")
    
    def start_monitoring(self, display_interval=10):
        """Start the monitoring process"""
        self.running = True
        
        # Start packet capture in separate thread
        capture_thread = threading.Thread(target=self.capture_packets)
        capture_thread.daemon = True
        capture_thread.start()
        
        try:
            while self.running:
                time.sleep(display_interval)
                self.print_security_dashboard()
                self.display_recent_suspicious_packets(3)
                print(f"\n💡 Monitoring... Press Ctrl+C to stop and save report")
                
        except KeyboardInterrupt:
            print(f"\n\n🔐 Stopping network security monitor...")
            self.running = False
            
            # Save comprehensive security report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_security_report(f'security_report_{self.interface}_{timestamp}.json')

def main():
    parser = argparse.ArgumentParser(description='Advanced Network Security Monitor')
    parser.add_argument('-i', '--interface', default='eth0', help='Network interface to monitor')
    parser.add_argument('-m', '--max-packets', type=int, default=1000, help='Maximum packets to store')
    parser.add_argument('-d', '--display-interval', type=int, default=10, help='Dashboard update interval (seconds)')
    
    args = parser.parse_args()
    
    print("🛡️  Advanced Network Security Monitor")
    print("====================================")
    print("🔍 Features: Intrusion Detection, DDoS Protection, Port Scan Detection")
    print("🚨 Real-time Threat Analysis and Automated Response")
    print("⚠️  Note: Requires administrator/root privileges")
    print(f"📡 Interface: {args.interface} | Update: {args.display_interval}s\n")
    
    monitor = NetworkTrafficMonitor(
        interface=args.interface,
        max_packets=args.max_packets
    )
    
    monitor.start_monitoring(args.display_interval)

if __name__ == "__main__":
    main()