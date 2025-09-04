#!/usr/bin/env python3
"""
Advanced Real-Time Threat Detection Backend with Live Packet Capture
Multi-layered security system with ML-powered threat detection
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import logging
import uuid
import hashlib
import re
import ipaddress
import sys
import os
import signal

# Web framework and async support
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
import requests
from concurrent.futures import ThreadPoolExecutor

# ML and data analysis
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Network packet capture and analysis
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, get_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scapy not installed. Install with: pip install scapy")
    print("   For Windows: may also need WinPcap or Npcap")
    print("   For Linux: may need to run as root for packet capture")
    SCAPY_AVAILABLE = False

# Network and security
import socket
import struct

@dataclass
class ThreatEvent:
    """Threat event data structure"""
    id: str
    timestamp: datetime
    source_ip: str
    destination_ip: str
    threat_type: str
    severity: int  # 1-10 scale
    confidence: float  # 0.0-1.0
    description: str
    indicators: List[str]
    raw_data: Dict[str, Any]
    blocked: bool = False
    false_positive: bool = False

class NetworkPacketCapture:
    """Live network packet capture and parsing"""
    
    def __init__(self, interface=None, filter_str=None):
        self.interface = interface
        self.filter_str = filter_str or "ip"
        self.capture_active = False
        self.packet_queue = asyncio.Queue()
        self.capture_thread = None
        self.packet_count = 0
        self.bytes_captured = 0
        
        # Get available interfaces
        if SCAPY_AVAILABLE:
            self.available_interfaces = get_if_list()
            print(f"📡 Available network interfaces: {self.available_interfaces}")
            
            # Auto-select interface if not specified
            if not self.interface:
                # Try to find a good default interface
                for iface in self.available_interfaces:
                    if 'eth' in iface.lower() or 'en' in iface.lower() or 'wlan' in iface.lower():
                        self.interface = iface
                        break
                else:
                    self.interface = self.available_interfaces[0] if self.available_interfaces else None
            
            print(f"🔍 Using interface: {self.interface}")
        else:
            print("❌ Scapy not available - packet capture disabled")
    
    def packet_handler(self, packet):
        """Handle captured packets"""
        try:
            if not packet.haslayer(IP):
                return
                
            ip_layer = packet[IP]
            packet_data = {
                'timestamp': datetime.now(),
                'source_ip': ip_layer.src,
                'destination_ip': ip_layer.dst,
                'protocol': ip_layer.proto,
                'packet_size': len(packet),
                'ttl': ip_layer.ttl,
                'flags': getattr(ip_layer, 'flags', 0),
                'id': ip_layer.id
            }
            
            # Parse protocol-specific data
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                packet_data.update({
                    'protocol': 'TCP',
                    'source_port': tcp_layer.sport,
                    'destination_port': tcp_layer.dport,
                    'tcp_flags': tcp_layer.flags,
                    'tcp_window': tcp_layer.window,
                    'tcp_seq': tcp_layer.seq,
                    'tcp_ack': tcp_layer.ack
                })
                
                # Extract payload
                if packet.haslayer(Raw):
                    packet_data['payload'] = bytes(packet[Raw])
                else:
                    packet_data['payload'] = b''
                    
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                packet_data.update({
                    'protocol': 'UDP',
                    'source_port': udp_layer.sport,
                    'destination_port': udp_layer.dport,
                    'udp_length': udp_layer.len
                })
                
                if packet.haslayer(Raw):
                    packet_data['payload'] = bytes(packet[Raw])
                else:
                    packet_data['payload'] = b''
                    
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                packet_data.update({
                    'protocol': 'ICMP',
                    'icmp_type': icmp_layer.type,
                    'icmp_code': icmp_layer.code,
                    'source_port': 0,
                    'destination_port': 0
                })
                
                if packet.haslayer(Raw):
                    packet_data['payload'] = bytes(packet[Raw])
                else:
                    packet_data['payload'] = b''
            
            # Calculate additional features
            packet_data.update({
                'bytes_transferred': len(packet_data.get('payload', b'')),
                'packets_per_second': 1,  # Will be calculated in behavioral analysis
                'unique_ports': 1,
                'protocol_diversity': 1,
                'time_between_connections': 0,
                'connection_duration': 0
            })
            
            # Add to queue for processing
            if not self.packet_queue.full():
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.packet_queue.put(packet_data))
                    )
                except:
                    # Fallback for queue full or loop issues
                    pass
            
            # Update statistics
            self.packet_count += 1
            self.bytes_captured += packet_data['packet_size']
            
            # Log periodic stats
            if self.packet_count % 1000 == 0:
                print(f"📊 Captured {self.packet_count} packets, {self.bytes_captured:,} bytes")
                
        except Exception as e:
            logging.error(f"Error processing packet: {e}")
    
    def start_capture(self):
        """Start packet capture in separate thread"""
        if not SCAPY_AVAILABLE:
            print("❌ Cannot start capture - scapy not available")
            return False
            
        if self.capture_active:
            print("⚠️  Capture already active")
            return True
            
        try:
            self.capture_active = True
            print(f"🚀 Starting packet capture on {self.interface}")
            print(f"   Filter: {self.filter_str}")
            print("   Press Ctrl+C to stop")
            
            # Start capture in separate thread
            def capture_worker():
                try:
                    sniff(
                        iface=self.interface,
                        filter=self.filter_str,
                        prn=self.packet_handler,
                        store=False,
                        stop_filter=lambda p: not self.capture_active
                    )
                except Exception as e:
                    logging.error(f"Packet capture error: {e}")
                    self.capture_active = False
            
            self.capture_thread = threading.Thread(target=capture_worker, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start packet capture: {e}")
            self.capture_active = False
            return False
    
    def stop_capture(self):
        """Stop packet capture"""
        print("🛑 Stopping packet capture...")
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        print(f"✅ Capture stopped. Total: {self.packet_count} packets, {self.bytes_captured:,} bytes")
    
    async def get_packet(self):
        """Get next packet from queue"""
        try:
            return await asyncio.wait_for(self.packet_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

class ThreatIntelligence:
    """Threat intelligence and reputation system"""
    
    def __init__(self):
        self.malicious_ips = set()
        self.suspicious_domains = set()
        self.known_malware_hashes = set()
        self.threat_feeds = []
        self.reputation_cache = {}
        self.last_update = None
        
        # Initialize with some basic threat data
        self.load_default_threats()
    
    def load_default_threats(self):
        """Load default threat intelligence"""
        # Known malicious IP ranges (examples - in real use, load from feeds)
        self.malicious_networks = [
            # Private ranges marked as suspicious for demo
            # In production, these would be actual threat IPs
        ]
        
        # Add some real suspicious IPs for demo
        self.malicious_ips.update([
            '10.0.0.1',  # Example suspicious IPs
            '192.168.1.1'
        ])
        
        # Known malicious domains
        self.suspicious_domains.update([
            'malware.com', 'phishing.net', 'botnet.org',
            'c2server.evil', 'trojan.download'
        ])
        
        # Sample malware hashes
        self.known_malware_hashes.update([
            'a1b2c3d4e5f6', 'deadbeefcafe', 'malware123456'
        ])
    
    def check_ip_reputation(self, ip: str) -> Tuple[bool, str, float]:
        """Check IP reputation"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            # Check against malicious networks
            for network in self.malicious_networks:
                if ip_addr in network:
                    return True, f"IP in known malicious network {network}", 0.9
            
            # Check direct IP matches
            if ip in self.malicious_ips:
                return True, "IP in threat intelligence feed", 0.95
                
            # Check for private/internal IPs doing suspicious things
            if ip_addr.is_private and (
                str(ip_addr).endswith('.1') or  # Gateways
                str(ip_addr).endswith('.255')   # Broadcast
            ):
                return True, f"Suspicious activity from internal IP {ip}", 0.6
                
            # Simulate external reputation check
            reputation_score = self.get_reputation_score(ip)
            if reputation_score < 0.3:
                return True, f"Poor reputation score: {reputation_score}", reputation_score
                
        except ValueError:
            pass
            
        return False, "Clean", 1.0
    
    def get_reputation_score(self, ip: str) -> float:
        """Simulate reputation scoring"""
        # In real implementation, query external APIs like VirusTotal, AbuseIPDB
        if ip in self.reputation_cache:
            return self.reputation_cache[ip]
        
        # Simulate scoring based on IP characteristics
        octets = ip.split('.')
        if len(octets) == 4:
            try:
                # Create some variance in scoring
                score = (int(octets[0]) + int(octets[3])) / 510.0
                # Add some randomness to make it more realistic
                import random
                score += random.uniform(-0.2, 0.2)
                score = max(0.0, min(1.0, score))
                self.reputation_cache[ip] = score
                return score
            except ValueError:
                pass
        
        return 0.5  # Neutral score

class BehavioralAnalyzer:
    """ML-powered behavioral analysis engine"""
    
    def __init__(self):
        self.connection_patterns = defaultdict(list)
        self.traffic_features = deque(maxlen=1000)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        self.baseline_established = False
        self.ip_stats = defaultdict(lambda: {'packets': 0, 'bytes': 0, 'ports': set(), 'first_seen': None})
        
    def extract_features(self, event_data: Dict) -> List[float]:
        """Extract features for ML analysis"""
        source_ip = event_data.get('source_ip', '')
        
        # Update IP statistics
        stats = self.ip_stats[source_ip]
        stats['packets'] += 1
        stats['bytes'] += event_data.get('packet_size', 0)
        if event_data.get('destination_port'):
            stats['ports'].add(event_data['destination_port'])
        if not stats['first_seen']:
            stats['first_seen'] = datetime.now()
        
        features = [
            event_data.get('packet_size', 0),
            event_data.get('connection_duration', 0),
            event_data.get('bytes_transferred', 0),
            event_data.get('packets_per_second', 0),
            len(stats['ports']),  # unique_ports for this IP
            event_data.get('protocol_diversity', 0),
            event_data.get('time_between_connections', 0),
            hash(source_ip) % 1000,  # IP hash feature
            len(event_data.get('payload', b'')),
            event_data.get('ttl', 0),  # TTL can indicate OS and distance
            event_data.get('tcp_window', 0) if 'tcp_window' in event_data else 0,
            stats['packets'],  # Total packets from this IP
            stats['bytes'],    # Total bytes from this IP
        ]
        return features
    
    def analyze_behavior(self, source_ip: str, event_data: Dict) -> Tuple[bool, float, str]:
        """Analyze behavioral patterns"""
        features = self.extract_features(event_data)
        self.traffic_features.append(features)
        
        # Store connection patterns
        self.connection_patterns[source_ip].append({
            'timestamp': datetime.now(),
            'features': features
        })
        
        # Train model if we have enough data
        if len(self.traffic_features) >= 100 and not self.model_trained:
            self.train_anomaly_detection()
        
        # Perform anomaly detection
        if self.model_trained:
            try:
                scaled_features = self.scaler.transform([features])
                anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
                is_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
                
                if is_anomaly:
                    return True, abs(anomaly_score), f"ML anomaly detected (score: {anomaly_score:.3f})"
            except Exception as e:
                logging.error(f"ML analysis error: {e}")
        
        # Rule-based behavioral analysis
        return self.rule_based_behavior_analysis(source_ip, event_data)
    
    def train_anomaly_detection(self):
        """Train the anomaly detection model"""
        try:
            features_array = np.array(list(self.traffic_features))
            scaled_features = self.scaler.fit_transform(features_array)
            self.isolation_forest.fit(scaled_features)
            self.model_trained = True
            logging.info("🤖 Anomaly detection model trained successfully")
        except Exception as e:
            logging.error(f"Failed to train anomaly detection model: {e}")
    
    def rule_based_behavior_analysis(self, source_ip: str, event_data: Dict) -> Tuple[bool, float, str]:
        """Rule-based behavioral analysis"""
        # Rapid connection analysis
        recent_connections = [
            conn for conn in self.connection_patterns[source_ip]
            if (datetime.now() - conn['timestamp']).seconds < 60
        ]
        
        if len(recent_connections) > 50:
            return True, 0.8, f"Rapid connection pattern: {len(recent_connections)} connections in 1 minute"
        
        # Port scanning detection
        stats = self.ip_stats[source_ip]
        if len(stats['ports']) > 20:
            return True, 0.9, f"Port scanning behavior: {len(stats['ports'])} unique ports"
        
        # High packet rate detection
        if stats['packets'] > 100:
            time_active = (datetime.now() - stats['first_seen']).total_seconds()
            if time_active > 0:
                packet_rate = stats['packets'] / time_active
                if packet_rate > 10:  # More than 10 packets per second
                    return True, 0.7, f"High packet rate: {packet_rate:.1f} packets/sec"
        
        # Large payload detection
        payload_size = len(event_data.get('payload', b''))
        if payload_size > 8192:  # Large payload
            return True, 0.6, f"Large payload detected: {payload_size} bytes"
        
        return False, 0.1, "Normal behavior"

class SignatureEngine:
    """Signature-based threat detection"""
    
    def __init__(self):
        self.malware_signatures = self.load_malware_signatures()
        self.attack_patterns = self.load_attack_patterns()
        self.regex_patterns = self.compile_regex_patterns()
    
    def load_malware_signatures(self) -> Dict[str, Dict]:
        """Load malware signatures"""
        return {
            'trojan_generic': {
                'patterns': [b'trojan', b'backdoor', b'keylogger'],
                'severity': 9,
                'description': 'Generic trojan signature'
            },
            'ransomware_pattern': {
                'patterns': [b'encrypt', b'ransom', b'bitcoin'],
                'severity': 10,
                'description': 'Ransomware activity detected'
            },
            'botnet_c2': {
                'patterns': [b'bot', b'command', b'control'],
                'severity': 8,
                'description': 'Botnet command and control'
            }
        }
    
    def load_attack_patterns(self) -> Dict[str, Dict]:
        """Load attack patterns"""
        return {
            'sql_injection': {
                'patterns': [b'union select', b'drop table', b"'; exec", b'xp_cmdshell'],
                'severity': 7,
                'description': 'SQL injection attempt'
            },
            'xss_attack': {
                'patterns': [b'<script>', b'javascript:', b'onerror=', b'onload='],
                'severity': 6,
                'description': 'Cross-site scripting attempt'
            },
            'command_injection': {
                'patterns': [b';cat ', b'|nc ', b'&& ', b'|| ', b'`whoami`'],
                'severity': 8,
                'description': 'Command injection attempt'
            },
            'directory_traversal': {
                'patterns': [b'../', b'..\\', b'etc/passwd', b'boot.ini'],
                'severity': 7,
                'description': 'Directory traversal attempt'
            },
            'password_attack': {
                'patterns': [b'admin', b'password', b'123456', b'root', b'login'],
                'severity': 5,
                'description': 'Potential password attack'
            }
        }
    
    def compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for faster matching"""
        patterns = {}
        
        # Suspicious URLs
        patterns['malicious_url'] = re.compile(
            rb'(malware|phishing|trojan|exploit|payload)\.(?:com|net|org)',
            re.IGNORECASE
        )
        
        # Suspicious file extensions
        patterns['malicious_file'] = re.compile(
            rb'\.(?:exe|scr|bat|cmd|pif|com|vbs|jar|app)$',
            re.IGNORECASE
        )
        
        # Base64 encoded payloads
        patterns['base64_payload'] = re.compile(
            rb'[A-Za-z0-9+/]{100,}={0,2}',
        )
        
        # Credit card numbers (PCI DSS concern)
        patterns['credit_card'] = re.compile(
            rb'\b(?:\d[ -]*?){13,16}\b'
        )
        
        return patterns
    
    def analyze_payload(self, payload: bytes) -> List[Tuple[str, int, str]]:
        """Analyze payload for signatures"""
        threats = []
        if not payload:
            return threats
            
        payload_lower = payload.lower()
        
        # Check malware signatures
        for sig_name, sig_data in self.malware_signatures.items():
            for pattern in sig_data['patterns']:
                if pattern in payload_lower:
                    threats.append((
                        'malware_signature',
                        sig_data['severity'],
                        f"{sig_data['description']}: {pattern.decode('utf-8', errors='ignore')}"
                    ))
        
        # Check attack patterns
        for attack_name, attack_data in self.attack_patterns.items():
            for pattern in attack_data['patterns']:
                if pattern in payload_lower:
                    threats.append((
                        attack_name,
                        attack_data['severity'],
                        f"{attack_data['description']}: {pattern.decode('utf-8', errors='ignore')}"
                    ))
        
        # Check regex patterns
        for pattern_name, regex in self.regex_patterns.items():
            if regex.search(payload):
                threats.append((
                    pattern_name,
                    6,
                    f"Suspicious {pattern_name.replace('_', ' ')} detected"
                ))
        
        return threats

class RealTimeThreatDetector:
    """Main threat detection engine with live packet capture"""
    
    def __init__(self, interface=None):
        self.threat_intel = ThreatIntelligence()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.signature_engine = SignatureEngine()
        self.packet_capture = NetworkPacketCapture(interface)
        
        # Threat storage and statistics
        self.active_threats = {}
        self.threat_history = deque(maxlen=10000)
        self.threat_stats = defaultdict(int)
        self.blocked_ips = set()
        
        # Real-time processing
        self.processing_active = False
        self.processing_task = None
        
        # Configuration
        self.auto_block_threshold = 8  # Auto-block threats with severity >= 8
        self.correlation_window = 300  # 5 minutes
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_live_monitoring(self):
        """Start live packet capture and threat detection"""
        if not SCAPY_AVAILABLE:
            self.logger.error("Cannot start live monitoring - scapy not available")
            return False
            
        print("🚀 Starting live threat detection...")
        
        # Start packet capture
        if not self.packet_capture.start_capture():
            self.logger.error("Failed to start packet capture")
            return False
        
        # Start processing loop
        self.processing_active = True
        self.processing_task = asyncio.create_task(self.packet_processing_loop())
        
        return True
    
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        print("🛑 Stopping live threat detection...")
        self.processing_active = False
        self.packet_capture.stop_capture()
        
        if self.processing_task:
            self.processing_task.cancel()
    
    async def packet_processing_loop(self):
        """Main packet processing loop"""
        print("🔄 Starting packet processing loop...")
        
        while self.processing_active:
            try:
                # Get packet from capture queue
                packet_data = await self.packet_capture.get_packet()
                if packet_data:
                    # Process packet for threats
                    threat_event = await self.process_network_event(packet_data)
                    
                    if threat_event:
                        # Emit real-time threat notification via WebSocket
                        try:
                            socketio.emit('threat_detected', {
                                'id': threat_event.id,
                                'timestamp': threat_event.timestamp.isoformat(),
                                'source_ip': threat_event.source_ip,
                                'destination_ip': threat_event.destination_ip,
                                'threat_type': threat_event.threat_type,
                                'severity': threat_event.severity,
                                'confidence': threat_event.confidence,
                                'description': threat_event.description,
                                'indicators': threat_event.indicators,
                                'blocked': threat_event.blocked
                            })
                        except Exception as e:
                            self.logger.error(f"Error emitting threat: {e}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in packet processing loop: {e}")
                await asyncio.sleep(1)
        
        print("✅ Packet processing loop stopped")
    
    async def process_network_event(self, event_data: Dict) -> Optional[ThreatEvent]:
        """Process network event for threats"""
        threats_detected = []
        max_severity = 0
        all_indicators = []
        
        source_ip = event_data.get('source_ip', '')
        dest_ip = event_data.get('destination_ip', '')
        payload = event_data.get('payload', b'')
        
        # Skip processing for certain IPs to reduce noise
        if self.should_skip_ip(source_ip):
            return None
        
        # 1. IP Reputation Check
        is_malicious, reputation_msg, confidence = self.threat_intel.check_ip_reputation(source_ip)
        if is_malicious:
            threats_detected.append(('ip_reputation', 7, reputation_msg))
            all_indicators.append(f"Malicious IP: {source_ip}")
        
        # 2. Signature-based Detection
        signature_threats = self.signature_engine.analyze_payload(payload)
        threats_detected.extend(signature_threats)
        
        # 3. Behavioral Analysis
        is_anomaly, behavior_confidence, behavior_msg = self.behavioral_analyzer.analyze_behavior(
            source_ip, event_data
        )
        if is_anomaly:
            severity = min(9, int(behavior_confidence * 10))
            threats_detected.append(('behavioral_anomaly', severity, behavior_msg))
            all_indicators.append(f"Behavioral anomaly: {behavior_confidence:.2f}")
        
        # 4. Protocol-specific analysis
        protocol_threats = self.analyze_protocol_specific(event_data)
        threats_detected.extend(protocol_threats)
        
        # Create threat event if any threats detected
        if threats_detected:
            max_severity = max(threat[1] for threat in threats_detected)
            threat_descriptions = [threat[2] for threat in threats_detected]
            
            threat_event = ThreatEvent(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                source_ip=source_ip,
                destination_ip=dest_ip,
                threat_type=threats_detected[0][0],
                severity=max_severity,
                confidence=max(confidence, behavior_confidence if is_anomaly else 0.5),
                description="; ".join(threat_descriptions[:3]),  # Limit description length
                indicators=all_indicators,
                raw_data=event_data
            )
            
            # Auto-block high severity threats
            if max_severity >= self.auto_block_threshold:
                threat_event.blocked = True
                self.blocked_ips.add(source_ip)
                self.logger.critical(f"🚫 AUTO-BLOCKED: {source_ip} - Severity {max_severity}")
            
            # Store and update statistics
            self.active_threats[threat_event.id] = threat_event
            self.threat_history.append(threat_event)
            self.threat_stats['total'] += 1
            self.threat_stats[threat_event.threat_type] += 1
            self.threat_stats[f'severity_{max_severity}'] += 1
            
            self.logger.warning(
                f"⚠️  THREAT: {source_ip} -> {dest_ip} "
                f"[{threat_event.threat_type}] Severity: {max_severity}"
            )
            
            return threat_event
        
        return None
    
    def should_skip_ip(self, ip: str) -> bool:
        """Check if IP should be skipped for processing"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            # Skip localhost and broadcast
            if ip_addr.is_loopback or str(ip) in ['255.255.255.255', '0.0.0.0']:
                return True
                
            # Skip multicast
            if ip_addr.is_multicast:
                return True
                
            return False
            
        except ValueError:
            return True  # Skip invalid IPs
    
    def analyze_protocol_specific(self, event_data: Dict) -> List[Tuple[str, int, str]]:
        """Protocol-specific threat analysis"""
        threats = []
        protocol = event_data.get('protocol', '').upper()
        
        if protocol == 'TCP':
            # TCP-specific analysis
            flags = event_data.get('tcp_flags', 0)
            src_port = event_data.get('source_port', 0)
            dst_port = event_data.get('destination_port', 0)
            
            # Suspicious TCP flags
            if flags == 0:  # NULL scan
                threats.append(('tcp_null_scan', 6, 'TCP NULL scan detected'))
            elif flags == 41:  # FIN, URG, PSH (XMAS scan)
                threats.append(('tcp_xmas_scan', 6, 'TCP XMAS scan detected'))
            
            # Suspicious ports
            suspicious_ports = [1337, 31337, 12345, 54321, 4444, 5555, 6666, 9999]
            if dst_port in suspicious_ports:
                threats.append(('suspicious_port', 7, f'Connection to suspicious port {dst_port}'))
            
            # Check for common malware ports
            malware_ports = {
                1337: 'Elite/Back Orifice',
                31337: 'Back Orifice',
                12345: 'NetBus',
                20034: 'NetBus Pro',
                9999: 'The Prayer trojan'
            }
            if dst_port in malware_ports:
                threats.append(('malware_port', 8, f'Connection to {malware_ports[dst_port]} port'))
        
        elif protocol == 'UDP':
            dst_port = event_data.get('destination_port', 0)
            packet_size = event_data.get('packet_size', 0)
            
            # DNS tunneling detection
            if dst_port == 53 and packet_size > 512:
                threats.append(('dns_tunneling', 8, f'Possible DNS tunneling: {packet_size} bytes'))
            
            # Check for UDP port scanning
            if dst_port in range(1, 1024) and packet_size < 64:
                threats.append(('udp_scan', 5, f'Possible UDP port scan on port {dst_port}'))
        
        elif protocol == 'ICMP':
            icmp_type = event_data.get('icmp_type', 0)
            packet_size = event_data.get('packet_size', 0)
            
            # Large ICMP packets (possible data exfiltration)
            if packet_size > 1000:
                threats.append(('icmp_exfiltration', 7, f'Large ICMP packet: {packet_size} bytes'))
            
            # ICMP tunneling detection
            if icmp_type == 8 and packet_size > 64:  # Echo request with large payload
                threats.append(('icmp_tunneling', 6, f'Possible ICMP tunneling: {packet_size} bytes'))
        
        return threats
    
    def correlate_threats(self) -> List[Dict]:
        """Correlate related threats"""
        correlations = []
        now = datetime.now()
        window_start = now - timedelta(seconds=self.correlation_window)
        
        # Get recent threats
        recent_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= window_start
        ]
        
        # Group by source IP
        ip_threats = defaultdict(list)
        for threat in recent_threats:
            ip_threats[threat.source_ip].append(threat)
        
        # Look for patterns
        for source_ip, threats in ip_threats.items():
            if len(threats) >= 3:  # Multiple threats from same IP
                correlations.append({
                    'type': 'coordinated_attack',
                    'source_ip': source_ip,
                    'threat_count': len(threats),
                    'severity': max(t.severity for t in threats),
                    'timespan': (threats[-1].timestamp - threats[0].timestamp).seconds,
                    'description': f'Coordinated attack from {source_ip}: {len(threats)} threats'
                })
        
        return correlations
    
    def get_threat_statistics(self) -> Dict:
        """Get comprehensive threat statistics"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        recent_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= hour_ago
        ]
        
        daily_threats = [
            threat for threat in self.threat_history
            if threat.timestamp >= day_ago
        ]
        
        return {
            'total_threats': len(self.threat_history),
            'active_threats': len(self.active_threats),
            'blocked_ips': len(self.blocked_ips),
            'threats_last_hour': len(recent_threats),
            'threats_last_24h': len(daily_threats),
            'threat_types': dict(self.threat_stats),
            'top_threat_ips': self.get_top_threat_ips(10),
            'severity_distribution': self.get_severity_distribution(),
            'correlations': self.correlate_threats(),
            'capture_stats': {
                'packets_captured': self.packet_capture.packet_count,
                'bytes_captured': self.packet_capture.bytes_captured,
                'capture_active': self.packet_capture.capture_active,
                'interface': self.packet_capture.interface
            }
        }
    
    def get_top_threat_ips(self, limit: int = 10) -> List[Dict]:
        """Get top threatening IP addresses"""
        ip_counts = defaultdict(int)
        ip_severity = defaultdict(int)
        
        for threat in self.threat_history:
            ip_counts[threat.source_ip] += 1
            ip_severity[threat.source_ip] = max(
                ip_severity[threat.source_ip],
                threat.severity
            )
        
        top_ips = sorted(
            ip_counts.items(),
            key=lambda x: (x[1], ip_severity[x[0]]),
            reverse=True
        )[:limit]
        
        return [
            {
                'ip': ip,
                'threat_count': count,
                'max_severity': ip_severity[ip],
                'blocked': ip in self.blocked_ips
            }
            for ip, count in top_ips
        ]
    
    def get_severity_distribution(self) -> Dict[str, int]:
        """Get threat severity distribution"""
        distribution = defaultdict(int)
        for threat in self.threat_history:
            severity_range = f"{(threat.severity // 2) * 2}-{(threat.severity // 2) * 2 + 1}"
            distribution[severity_range] += 1
        return dict(distribution)

# Flask Web API
app = Flask(__name__)
app.config['SECRET_KEY'] = 'threat_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global threat detector instance
threat_detector = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}")
    if threat_detector:
        threat_detector.stop_live_monitoring()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/api/threats', methods=['GET'])
def get_threats():
    """Get recent threats"""
    if not threat_detector:
        return jsonify({'error': 'Threat detector not initialized'}), 500
        
    limit = request.args.get('limit', 100, type=int)
    recent_threats = list(threat_detector.threat_history)[-limit:]
    
    return jsonify([
        {
            'id': threat.id,
            'timestamp': threat.timestamp.isoformat(),
            'source_ip': threat.source_ip,
            'destination_ip': threat.destination_ip,
            'threat_type': threat.threat_type,
            'severity': threat.severity,
            'confidence': threat.confidence,
            'description': threat.description,
            'indicators': threat.indicators,
            'blocked': threat.blocked
        }
        for threat in recent_threats
    ])

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get threat statistics"""
    if not threat_detector:
        return jsonify({'error': 'Threat detector not initialized'}), 500
        
    return jsonify(threat_detector.get_threat_statistics())

@app.route('/api/block/<ip>', methods=['POST'])
def block_ip(ip):
    """Manually block an IP"""
    if not threat_detector:
        return jsonify({'error': 'Threat detector not initialized'}), 500
        
    threat_detector.blocked_ips.add(ip)
    threat_detector.logger.info(f"Manually blocked IP: {ip}")
    
    # Emit real-time update
    socketio.emit('ip_blocked', {'ip': ip, 'manual': True})
    
    return jsonify({'status': 'success', 'message': f'IP {ip} blocked'})

@app.route('/api/unblock/<ip>', methods=['POST'])
def unblock_ip(ip):
    """Unblock an IP"""
    if not threat_detector:
        return jsonify({'error': 'Threat detector not initialized'}), 500
        
    threat_detector.blocked_ips.discard(ip)
    threat_detector.logger.info(f"Unblocked IP: {ip}")
    
    # Emit real-time update
    socketio.emit('ip_unblocked', {'ip': ip})
    
    return jsonify({'status': 'success', 'message': f'IP {ip} unblocked'})

@app.route('/api/interfaces', methods=['GET'])
def get_interfaces():
    """Get available network interfaces"""
    if SCAPY_AVAILABLE:
        try:
            from scapy.all import get_if_list
            interfaces = get_if_list()
            return jsonify({'interfaces': interfaces})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Scapy not available'}), 500

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start live monitoring"""
    global threat_detector
    
    if not SCAPY_AVAILABLE:
        return jsonify({'error': 'Scapy not available - cannot start monitoring'}), 500
    
    data = request.json or {}
    interface = data.get('interface')
    
    try:
        if not threat_detector:
            threat_detector = RealTimeThreatDetector(interface)
        
        # Start monitoring in background
        def start_async_monitoring():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(threat_detector.start_live_monitoring())
            except Exception as e:
                logging.error(f"Error starting monitoring: {e}")
            finally:
                loop.close()
        
        monitoring_thread = threading.Thread(target=start_async_monitoring, daemon=True)
        monitoring_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Live monitoring started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop live monitoring"""
    if threat_detector:
        threat_detector.stop_live_monitoring()
        return jsonify({'status': 'success', 'message': 'Live monitoring stopped'})
    else:
        return jsonify({'error': 'No active monitoring'}), 400

@app.route('/api/submit_event', methods=['POST'])
def submit_event():
    """Submit network event for manual analysis"""
    if not threat_detector:
        return jsonify({'error': 'Threat detector not initialized'}), 500
        
    event_data = request.json
    
    async def process_event():
        threat_event = await threat_detector.process_network_event(event_data)
        if threat_event:
            # Emit real-time threat notification
            socketio.emit('threat_detected', {
                'id': threat_event.id,
                'timestamp': threat_event.timestamp.isoformat(),
                'source_ip': threat_event.source_ip,
                'threat_type': threat_event.threat_type,
                'severity': threat_event.severity,
                'description': threat_event.description,
                'blocked': threat_event.blocked
            })
            
            return asdict(threat_event)
        return None
    
    # Run async processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_event())
        if result:
            return jsonify({'status': 'threat_detected', 'threat': result})
        else:
            return jsonify({'status': 'clean'})
    finally:
        loop.close()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'scapy_available': SCAPY_AVAILABLE,
    }
    
    if threat_detector:
        status.update({
            'active_threats': len(threat_detector.active_threats),
            'blocked_ips': len(threat_detector.blocked_ips),
            'monitoring_active': threat_detector.processing_active,
            'packets_captured': threat_detector.packet_capture.packet_count
        })
    else:
        status.update({
            'active_threats': 0,
            'blocked_ips': 0,
            'monitoring_active': False,
            'packets_captured': 0
        })
    
    return jsonify(status)

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('connected', {
        'status': 'Connected to live threat detection system',
        'scapy_available': SCAPY_AVAILABLE,
        'monitoring_active': threat_detector.processing_active if threat_detector else False
    })

@socketio.on('subscribe_threats')
def handle_subscribe():
    """Subscribe to real-time threat updates"""
    emit('subscribed', {'message': 'Subscribed to real-time threat updates'})

@socketio.on('get_status')
def handle_get_status():
    """Get current system status"""
    status = {
        'scapy_available': SCAPY_AVAILABLE,
        'monitoring_active': threat_detector.processing_active if threat_detector else False,
        'interface': threat_detector.packet_capture.interface if threat_detector else None,
        'packets_captured': threat_detector.packet_capture.packet_count if threat_detector else 0
    }
    emit('status_update', status)

def check_requirements():
    """Check if required dependencies are available"""
    missing = []
    
    if not SCAPY_AVAILABLE:
        missing.append("scapy")
    
    try:
        import numpy
        import sklearn
    except ImportError:
        missing.append("numpy, scikit-learn")
    
    try:
        from flask import Flask
        from flask_socketio import SocketIO
    except ImportError:
        missing.append("flask, flask-socketio")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   pip install {dep}")
        print("\nFor packet capture, you may also need:")
        print("   - Linux: Run as root or with CAP_NET_RAW capability")
        print("   - Windows: Install WinPcap or Npcap")
        print("   - macOS: May need to run with sudo")
        return False
    
    return True

if __name__ == '__main__':
    print("🛡️  Advanced Real-Time Threat Detection Backend")
    print("=" * 60)
    print("🔍 Multi-layered security analysis")
    print("🤖 ML-powered behavioral detection")
    print("📡 Real-time threat intelligence")
    print("📦 Live network packet capture")
    print("🌐 REST API and WebSocket support")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start - missing dependencies")
        sys.exit(1)
    
    if not SCAPY_AVAILABLE:
        print("⚠️  Warning: Scapy not available - live capture disabled")
        print("   Install with: pip install scapy")
    else:
        print("✅ Scapy available - live packet capture enabled")
        
        # Check for admin privileges
        if os.name == 'nt':  # Windows
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("⚠️  Warning: Not running as administrator - packet capture may fail")
        elif os.name == 'posix':  # Linux/macOS
            if os.geteuid() != 0:
                print("⚠️  Warning: Not running as root - packet capture may fail")
                print("   Try: sudo python3 threat_detector.py")
    
    print("\nAPI Endpoints:")
    print("  GET  /api/threats - Get recent threats")
    print("  GET  /api/statistics - Get threat statistics")
    print("  GET  /api/interfaces - Get network interfaces")
    print("  POST /api/start_monitoring - Start live monitoring")
    print("  POST /api/stop_monitoring - Stop live monitoring")
    print("  POST /api/block/<ip> - Block IP address")
    print("  POST /api/unblock/<ip> - Unblock IP address")
    print("  GET  /api/health - Health check")
    print("\nWebSocket Events:")
    print("  threat_detected - Real-time threat alerts")
    print("  ip_blocked/unblocked - IP block status changes")
    print("  status_update - System status updates")
    print()
    
    try:
        # Initialize the global threat detector
        threat_detector = RealTimeThreatDetector()
        
        print("🚀 Starting web server on http://0.0.0.0:5000")
        print("   Connect your frontend to this backend")
        print("   Use POST /api/start_monitoring to begin live capture")
        print("\n   Press Ctrl+C to stop")
        
        # Start the web server
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if threat_detector:
            threat_detector.stop_live_monitoring()
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        if threat_detector:
            threat_detector.stop_live_monitoring()
        sys.exit(1)