#!/usr/bin/env python3
"""
Advanced Machine Learning Threat Detection Model
Comprehensive ML system for cybersecurity threat classification using Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import re
import ipaddress
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class ThreatFeatures:
    """Comprehensive feature vector for threat classification"""
    # Network features
    packet_size: float
    connection_duration: float
    bytes_transferred: float
    packets_per_second: float
    unique_ports: int
    protocol_diversity: float
    
    # Behavioral features
    connection_frequency: float
    time_between_connections: float
    geographic_distance: float
    reputation_score: float
    
    # Content features
    payload_entropy: float
    suspicious_strings: int
    base64_content: bool
    encrypted_content: bool
    
    # Temporal features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    
    # Source features
    source_ip_class: int
    is_tor_exit: bool
    is_vpn: bool
    country_risk_score: float
    
    # Advanced features
    tcp_flags_anomaly: bool
    port_scan_indicator: bool
    ddos_indicator: bool
    malware_signature_count: int
    
    def to_vector(self) -> List[float]:
        """Convert features to numerical vector"""
        return [
            self.packet_size,
            self.connection_duration,
            self.bytes_transferred,
            self.packets_per_second,
            float(self.unique_ports),
            self.protocol_diversity,
            self.connection_frequency,
            self.time_between_connections,
            self.geographic_distance,
            self.reputation_score,
            self.payload_entropy,
            float(self.suspicious_strings),
            float(self.base64_content),
            float(self.encrypted_content),
            float(self.hour_of_day),
            float(self.day_of_week),
            float(self.is_weekend),
            float(self.source_ip_class),
            float(self.is_tor_exit),
            float(self.is_vpn),
            self.country_risk_score,
            float(self.tcp_flags_anomaly),
            float(self.port_scan_indicator),
            float(self.ddos_indicator),
            float(self.malware_signature_count)
        ]

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_samples: int
    test_samples: int
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    training_time: float
    last_trained: str

class ThreatDetectionModel:
    """Advanced ML-based threat detection system"""
    
    def __init__(self, model_dir: str = "ml_models/saved_models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.primary_classifier = None
        self.anomaly_detector = None
        self.feature_selector = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature names for interpretability
        self.feature_names = [
            'packet_size', 'connection_duration', 'bytes_transferred', 'packets_per_second',
            'unique_ports', 'protocol_diversity', 'connection_frequency', 'time_between_connections',
            'geographic_distance', 'reputation_score', 'payload_entropy', 'suspicious_strings',
            'base64_content', 'encrypted_content', 'hour_of_day', 'day_of_week', 'is_weekend',
            'source_ip_class', 'is_tor_exit', 'is_vpn', 'country_risk_score', 'tcp_flags_anomaly',
            'port_scan_indicator', 'ddos_indicator', 'malware_signature_count'
        ]
        
        # Threat categories
        self.threat_categories = [
            'benign', 'malware', 'botnet', 'ddos', 'brute_force',
            'sql_injection', 'xss', 'phishing', 'ransomware', 'apt',
            'port_scan', 'data_exfiltration', 'lateral_movement'
        ]
        
        # Model configuration
        self.model_config = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'isolation_forest': {
                'contamination': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Performance tracking
        self.metrics = None
        self.training_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Try to load existing models
        self.load_models()
    
    def extract_features(self, event_data: Dict) -> ThreatFeatures:
        """Extract comprehensive features from network event data"""
        # Network features
        packet_size = float(event_data.get('packet_size', 0))
        connection_duration = float(event_data.get('connection_duration', 0))
        bytes_transferred = float(event_data.get('bytes_transferred', packet_size))
        packets_per_second = float(event_data.get('packets_per_second', 1))
        unique_ports = int(event_data.get('unique_ports', 1))
        protocol_diversity = float(event_data.get('protocol_diversity', 0))
        
        # Behavioral features
        connection_frequency = float(event_data.get('connection_frequency', 0))
        time_between_connections = float(event_data.get('time_between_connections', 0))
        geographic_distance = float(event_data.get('geographic_distance', 0))
        reputation_score = float(event_data.get('reputation_score', 0.5))
        
        # Content analysis
        payload = event_data.get('payload', b'')
        if isinstance(payload, str):
            payload = payload.encode('utf-8', errors='ignore')
        
        payload_entropy = self.calculate_entropy(payload)
        suspicious_strings = self.count_suspicious_strings(payload)
        base64_content = self.detect_base64(payload)
        encrypted_content = self.detect_encryption(payload)
        
        # Temporal features
        timestamp = event_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        # Source analysis
        source_ip = event_data.get('source_ip', '0.0.0.0')
        source_ip_class = self.classify_ip(source_ip)
        is_tor_exit = event_data.get('is_tor_exit', False)
        is_vpn = event_data.get('is_vpn', False)
        country_risk_score = float(event_data.get('country_risk_score', 0.5))
        
        # Advanced threat indicators
        tcp_flags_anomaly = self.detect_tcp_flags_anomaly(event_data)
        port_scan_indicator = self.detect_port_scan_pattern(event_data)
        ddos_indicator = self.detect_ddos_pattern(event_data)
        malware_signature_count = self.count_malware_signatures(payload)
        
        return ThreatFeatures(
            packet_size=packet_size,
            connection_duration=connection_duration,
            bytes_transferred=bytes_transferred,
            packets_per_second=packets_per_second,
            unique_ports=unique_ports,
            protocol_diversity=protocol_diversity,
            connection_frequency=connection_frequency,
            time_between_connections=time_between_connections,
            geographic_distance=geographic_distance,
            reputation_score=reputation_score,
            payload_entropy=payload_entropy,
            suspicious_strings=suspicious_strings,
            base64_content=base64_content,
            encrypted_content=encrypted_content,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            source_ip_class=source_ip_class,
            is_tor_exit=is_tor_exit,
            is_vpn=is_vpn,
            country_risk_score=country_risk_score,
            tcp_flags_anomaly=tcp_flags_anomaly,
            port_scan_indicator=port_scan_indicator,
            ddos_indicator=ddos_indicator,
            malware_signature_count=malware_signature_count
        )
    
    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def count_suspicious_strings(self, payload: bytes) -> int:
        """Count suspicious strings in payload"""
        suspicious_patterns = [
            b'eval', b'exec', b'system', b'shell', b'cmd',
            b'script', b'javascript', b'vbscript',
            b'union', b'select', b'drop', b'insert', b'update',
            b'<script>', b'</script>', b'onerror', b'onload',
            b'../../../', b'..\\..\\..\\',
            b'passwd', b'shadow', b'hosts',
            b'trojan', b'backdoor', b'malware', b'virus',
            b'botnet', b'c2', b'command', b'control'
        ]
        
        count = 0
        payload_lower = payload.lower()
        
        for pattern in suspicious_patterns:
            count += payload_lower.count(pattern)
        
        return count
    
    def detect_base64(self, payload: bytes) -> bool:
        """Detect base64 encoded content"""
        try:
            base64_pattern = re.compile(rb'[A-Za-z0-9+/]{20,}={0,2}')
            matches = base64_pattern.findall(payload)
            return len(matches) > 0
        except:
            return False
    
    def detect_encryption(self, payload: bytes) -> bool:
        """Detect encrypted content"""
        if not payload:
            return False
        
        # High entropy suggests encryption
        entropy = self.calculate_entropy(payload)
        return entropy > 7.0
    
    def classify_ip(self, ip_str: str) -> int:
        """Classify IP address type"""
        try:
            ip = ipaddress.ip_address(ip_str)
            
            if ip.is_private:
                return 1  # Private
            elif ip.is_loopback:
                return 2  # Loopback
            elif ip.is_multicast:
                return 3  # Multicast
            elif ip.is_reserved:
                return 4  # Reserved
            else:
                return 5  # Public
        except:
            return 0  # Invalid
    
    def detect_tcp_flags_anomaly(self, event_data: Dict) -> bool:
        """Detect TCP flags anomalies"""
        tcp_flags = event_data.get('tcp_flags', 0)
        if tcp_flags == 0:  # NULL scan
            return True
        elif tcp_flags == 41:  # XMAS scan (FIN+URG+PSH)
            return True
        elif tcp_flags == 1:  # FIN scan
            return True
        return False
    
    def detect_port_scan_pattern(self, event_data: Dict) -> bool:
        """Detect port scanning patterns"""
        dest_port = event_data.get('destination_port', 0)
        packet_size = event_data.get('packet_size', 0)
        
        # Small packets to common ports might indicate scanning
        if packet_size < 100 and dest_port in range(1, 1024):
            return True
        
        # Check for sequential port access patterns
        unique_ports = event_data.get('unique_ports', 1)
        if unique_ports > 10:
            return True
        
        return False
    
    def detect_ddos_pattern(self, event_data: Dict) -> bool:
        """Detect DDoS attack patterns"""
        packets_per_second = event_data.get('packets_per_second', 0)
        bytes_transferred = event_data.get('bytes_transferred', 0)
        
        # High packet rate with small packets
        if packets_per_second > 100 and bytes_transferred / max(packets_per_second, 1) < 100:
            return True
        
        return False
    
    def count_malware_signatures(self, payload: bytes) -> int:
        """Count malware signatures in payload"""
        malware_patterns = [
            b'metasploit', b'meterpreter', b'payload', b'shellcode',
            b'backdoor', b'trojan', b'keylogger', b'rootkit',
            b'ransomware', b'cryptolocker', b'wannacry'
        ]
        
        count = 0
        payload_lower = payload.lower()
        
        for pattern in malware_patterns:
            if pattern in payload_lower:
                count += 1
        
        return count
    
    def generate_synthetic_training_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for model training"""
        self.logger.info(f"Generating {num_samples} synthetic training samples...")
        
        features = []
        labels = []
        
        for _ in range(num_samples):
            # Randomly select threat category
            threat_category = np.random.choice(self.threat_categories)
            
            # Generate features based on threat type
            if threat_category == 'benign':
                feature_vector = self._generate_benign_features()
            elif threat_category == 'malware':
                feature_vector = self._generate_malware_features()
            elif threat_category == 'ddos':
                feature_vector = self._generate_ddos_features()
            elif threat_category == 'port_scan':
                feature_vector = self._generate_port_scan_features()
            elif threat_category == 'brute_force':
                feature_vector = self._generate_brute_force_features()
            else:
                feature_vector = self._generate_generic_threat_features()
            
            features.append(feature_vector)
            labels.append(threat_category)
        
        return np.array(features), np.array(labels)
    
    def _generate_benign_features(self) -> List[float]:
        """Generate features for benign traffic"""
        return [
            np.random.normal(500, 200),  # packet_size
            np.random.exponential(10),   # connection_duration
            np.random.normal(5000, 2000), # bytes_transferred
            np.random.normal(5, 2),      # packets_per_second
            np.random.randint(1, 5),     # unique_ports
            np.random.uniform(0, 0.3),   # protocol_diversity
            np.random.normal(2, 1),      # connection_frequency
            np.random.exponential(30),   # time_between_connections
            np.random.uniform(0, 5000),  # geographic_distance
            np.random.uniform(0.7, 1.0), # reputation_score (high for benign)
            np.random.uniform(3, 6),     # payload_entropy (normal)
            np.random.randint(0, 2),     # suspicious_strings (few)
            False,                       # base64_content
            False,                       # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            False,                       # is_tor_exit
            False,                       # is_vpn
            np.random.uniform(0.8, 1.0), # country_risk_score (low risk)
            False,                       # tcp_flags_anomaly
            False,                       # port_scan_indicator
            False,                       # ddos_indicator
            0                            # malware_signature_count
        ]
    
    def _generate_malware_features(self) -> List[float]:
        """Generate features for malware traffic"""
        return [
            np.random.normal(800, 400),  # packet_size (larger)
            np.random.exponential(5),    # connection_duration (shorter)
            np.random.normal(15000, 5000), # bytes_transferred (more data)
            np.random.normal(15, 8),     # packets_per_second (higher)
            np.random.randint(3, 15),    # unique_ports (more ports)
            np.random.uniform(0.3, 0.8), # protocol_diversity
            np.random.normal(10, 5),     # connection_frequency (higher)
            np.random.exponential(5),    # time_between_connections (faster)
            np.random.uniform(1000, 10000), # geographic_distance
            np.random.uniform(0.1, 0.4), # reputation_score (low)
            np.random.uniform(6, 8),     # payload_entropy (high)
            np.random.randint(3, 10),    # suspicious_strings (many)
            np.random.choice([True, False]), # base64_content
            np.random.choice([True, False]), # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            np.random.choice([True, False]), # is_tor_exit
            np.random.choice([True, False]), # is_vpn
            np.random.uniform(0.1, 0.5), # country_risk_score (high risk)
            np.random.choice([True, False]), # tcp_flags_anomaly
            False,                       # port_scan_indicator
            False,                       # ddos_indicator
            np.random.randint(1, 5)      # malware_signature_count
        ]
    
    def _generate_ddos_features(self) -> List[float]:
        """Generate features for DDoS attacks"""
        return [
            np.random.normal(64, 20),    # packet_size (small packets)
            np.random.uniform(0.1, 1),   # connection_duration (very short)
            np.random.normal(100, 50),   # bytes_transferred (small)
            np.random.normal(500, 200),  # packets_per_second (very high)
            np.random.randint(1, 3),     # unique_ports (few ports)
            np.random.uniform(0, 0.2),   # protocol_diversity (low)
            np.random.normal(100, 50),   # connection_frequency (very high)
            np.random.uniform(0.01, 0.1), # time_between_connections (very fast)
            np.random.uniform(0, 2000),  # geographic_distance
            np.random.uniform(0.2, 0.6), # reputation_score
            np.random.uniform(1, 4),     # payload_entropy (low)
            np.random.randint(0, 2),     # suspicious_strings
            False,                       # base64_content
            False,                       # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            False,                       # is_tor_exit
            False,                       # is_vpn
            np.random.uniform(0.3, 0.7), # country_risk_score
            np.random.choice([True, False]), # tcp_flags_anomaly
            False,                       # port_scan_indicator
            True,                        # ddos_indicator
            0                            # malware_signature_count
        ]
    
    def _generate_port_scan_features(self) -> List[float]:
        """Generate features for port scanning"""
        return [
            np.random.normal(40, 15),    # packet_size (very small)
            np.random.uniform(0.1, 2),   # connection_duration (short)
            np.random.normal(50, 20),    # bytes_transferred (minimal)
            np.random.normal(20, 10),    # packets_per_second (moderate)
            np.random.randint(10, 50),   # unique_ports (many ports)
            np.random.uniform(0.1, 0.4), # protocol_diversity
            np.random.normal(30, 15),    # connection_frequency (high)
            np.random.uniform(0.1, 1),   # time_between_connections (fast)
            np.random.uniform(0, 3000),  # geographic_distance
            np.random.uniform(0.3, 0.7), # reputation_score
            np.random.uniform(1, 3),     # payload_entropy (very low)
            np.random.randint(0, 1),     # suspicious_strings
            False,                       # base64_content
            False,                       # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            False,                       # is_tor_exit
            False,                       # is_vpn
            np.random.uniform(0.4, 0.8), # country_risk_score
            np.random.choice([True, False]), # tcp_flags_anomaly
            True,                        # port_scan_indicator
            False,                       # ddos_indicator
            0                            # malware_signature_count
        ]
    
    def _generate_brute_force_features(self) -> List[float]:
        """Generate features for brute force attacks"""
        return [
            np.random.normal(200, 100),  # packet_size
            np.random.uniform(1, 5),     # connection_duration
            np.random.normal(1000, 500), # bytes_transferred
            np.random.normal(8, 4),      # packets_per_second
            np.random.randint(1, 3),     # unique_ports (few ports, usually 22, 3389, etc.)
            np.random.uniform(0, 0.2),   # protocol_diversity (low)
            np.random.normal(50, 20),    # connection_frequency (high)
            np.random.uniform(1, 10),    # time_between_connections
            np.random.uniform(0, 5000),  # geographic_distance
            np.random.uniform(0.2, 0.5), # reputation_score (poor)
            np.random.uniform(2, 5),     # payload_entropy
            np.random.randint(1, 5),     # suspicious_strings
            False,                       # base64_content
            False,                       # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            np.random.choice([True, False]), # is_tor_exit
            np.random.choice([True, False]), # is_vpn
            np.random.uniform(0.2, 0.6), # country_risk_score
            False,                       # tcp_flags_anomaly
            False,                       # port_scan_indicator
            False,                       # ddos_indicator
            np.random.randint(0, 2)      # malware_signature_count
        ]
    
    def _generate_generic_threat_features(self) -> List[float]:
        """Generate features for generic threats"""
        return [
            np.random.normal(600, 300),  # packet_size
            np.random.exponential(8),    # connection_duration
            np.random.normal(8000, 4000), # bytes_transferred
            np.random.normal(12, 6),     # packets_per_second
            np.random.randint(2, 10),    # unique_ports
            np.random.uniform(0.2, 0.6), # protocol_diversity
            np.random.normal(8, 4),      # connection_frequency
            np.random.exponential(10),   # time_between_connections
            np.random.uniform(500, 8000), # geographic_distance
            np.random.uniform(0.2, 0.6), # reputation_score
            np.random.uniform(4, 7),     # payload_entropy
            np.random.randint(1, 6),     # suspicious_strings
            np.random.choice([True, False]), # base64_content
            np.random.choice([True, False]), # encrypted_content
            np.random.randint(0, 24),    # hour_of_day
            np.random.randint(0, 7),     # day_of_week
            np.random.choice([True, False]), # is_weekend
            np.random.randint(1, 5),     # source_ip_class
            np.random.choice([True, False]), # is_tor_exit
            np.random.choice([True, False]), # is_vpn
            np.random.uniform(0.3, 0.7), # country_risk_score
            np.random.choice([True, False]), # tcp_flags_anomaly
            np.random.choice([True, False]), # port_scan_indicator
            np.random.choice([True, False]), # ddos_indicator
            np.random.randint(0, 3)      # malware_signature_count
        ]
    
    def train_model(self, training_data: Optional[List[Dict]] = None, labels: Optional[List[str]] = None):
        """Train the threat detection model"""
        start_time = datetime.now()
        self.logger.info("Starting model training...")
        
        # Use synthetic data if no training data provided
        if training_data is None or labels is None:
            X, y = self.generate_synthetic_training_data()
        else:
            # Extract features from provided training data
            feature_vectors = []
            for event in training_data:
                features = self.extract_features(event)
                feature_vectors.append(features.to_vector())
            
            X = np.array(feature_vectors)
            y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Hyperparameter tuning for Random Forest
        self.logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Use best parameters
        self.primary_classifier = grid_search.best_estimator_
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(**self.model_config['isolation_forest'])
        self.anomaly_detector.fit(X_train_scaled)
        
        # Evaluate model
        y_pred = self.primary_classifier.predict(X_test_scaled)
        y_pred_proba = self.primary_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC score (for binary classification, use macro average for multiclass)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # Feature importance
        feature_importance = {}
        selected_features = self.feature_selector.get_support()
        selected_feature_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
        
        for i, importance in enumerate(self.primary_classifier.feature_importances_):
            if i < len(selected_feature_names):
                feature_importance[selected_feature_names[i]] = float(importance)
        
        # Store metrics
        training_time = (datetime.now() - start_time).total_seconds()
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            training_samples=len(X_train),
            test_samples=len(X_test),
            feature_importance=feature_importance,
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            training_time=training_time,
            last_trained=datetime.now().isoformat()
        )
        
        # Log results
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        self.logger.info(f"Accuracy: {accuracy:.3f}")
        self.logger.info(f"F1 Score: {f1:.3f}")
        self.logger.info(f"AUC Score: {auc:.3f}")
        
        # Save models
        self.save_models()
        
        # Generate training report
        self.generate_training_report()
        
        return self.metrics
    
    def predict_threat(self, event_data: Dict) -> Dict[str, Any]:
        """Predict threat using trained models"""
        try:
            # Extract features
            features = self.extract_features(event_data)
            feature_vector = np.array([features.to_vector()])
            
            # Apply feature selection and scaling
            if self.feature_selector is not None:
                feature_vector = self.feature_selector.transform(feature_vector)
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_version': '2.0',
                'features_extracted': len(features.to_vector())
            }
            
            # Primary classification
            if self.primary_classifier is not None:
                prediction = self.primary_classifier.predict(feature_vector_scaled)[0]
                probabilities = self.primary_classifier.predict_proba(feature_vector_scaled)[0]
                
                threat_type = self.label_encoder.inverse_transform([prediction])[0]
                confidence = float(np.max(probabilities))
                
                results['primary_classification'] = {
                    'threat_type': threat_type,
                    'confidence': confidence,
                    'probabilities': {
                        category: float(prob) 
                        for category, prob in zip(
                            self.label_encoder.classes_, probabilities
                        )
                    }
                }
            
            # Anomaly detection
            if self.anomaly_detector is not None:
                anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(feature_vector_scaled)[0] == -1
                
                results['anomaly_detection'] = {
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'threshold': 0.0
                }
            
            # Rule-based classification
            rule_based = self.rule_based_classification(features)
            results['rule_based'] = rule_based
            
            # Combine results
            final_classification = self.combine_classifications(results)
            results['final_classification'] = final_classification
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.fallback_prediction(event_data)
    
    def rule_based_classification(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Rule-based threat classification"""
        rules_triggered = []
        severity = 1
        
        # Network anomalies
        if features.packets_per_second > 100:
            rules_triggered.append('high_packet_rate')
            severity = max(severity, 8)
        
        if features.bytes_transferred > 50_000_000:  # 50MB
            rules_triggered.append('large_data_transfer')
            severity = max(severity, 7)
        
        # Content analysis
        if features.suspicious_strings > 5:
            rules_triggered.append('suspicious_content')
            severity = max(severity, 9)
        
        if features.payload_entropy > 7.5:
            rules_triggered.append('high_entropy_payload')
            severity = max(severity, 7)
        
        # Behavioral analysis
        if features.connection_frequency > 50:
            rules_triggered.append('rapid_connections')
            severity = max(severity, 8)
        
        # Source reputation
        if features.reputation_score < 0.3:
            rules_triggered.append('poor_reputation')
            severity = max(severity, 8)
        
        # Advanced indicators
        if features.tcp_flags_anomaly:
            rules_triggered.append('tcp_flags_anomaly')
            severity = max(severity, 7)
        
        if features.port_scan_indicator:
            rules_triggered.append('port_scan_detected')
            severity = max(severity, 6)
        
        if features.ddos_indicator:
            rules_triggered.append('ddos_pattern')
            severity = max(severity, 9)
        
        if features.malware_signature_count > 0:
            rules_triggered.append('malware_signatures')
            severity = max(severity, 9)
        
        return {
            'rules_triggered': rules_triggered,
            'severity': severity,
            'is_threat': len(rules_triggered) > 0,
            'confidence': min(1.0, len(rules_triggered) * 0.2)
        }
    
    def combine_classifications(self, results: Dict) -> Dict[str, Any]:
        """Combine multiple classification results"""
        primary = results.get('primary_classification', {})
        anomaly = results.get('anomaly_detection', {})
        rule_based = results.get('rule_based', {})
        
        # Start with primary classification
        threat_type = primary.get('threat_type', 'unknown')
        confidence = primary.get('confidence', 0.5)
        severity = rule_based.get('severity', 1)
        
        # Adjust based on anomaly detection
        if anomaly.get('is_anomaly', False):
            confidence = min(1.0, confidence + 0.3)
            severity = max(severity, 7)
            if threat_type == 'benign':
                threat_type = 'anomaly'
        
        # Adjust based on rule-based results
        if rule_based.get('is_threat', False):
            confidence = min(1.0, confidence + 0.2)
            severity = max(severity, rule_based.get('severity', 1))
            
            # Override benign classification if rules detect threats
            if threat_type == 'benign':
                threat_type = 'malware'  # Generic threat type
        
        return {
            'threat_type': threat_type,
            'confidence': float(confidence),
            'severity': int(severity),
            'is_threat': threat_type != 'benign' and threat_type != 'unknown',
            'classification_method': 'ensemble'
        }
    
    def fallback_prediction(self, event_data: Dict) -> Dict[str, Any]:
        """Fallback prediction when models fail"""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_version': '2.0',
            'features_extracted': 0,
            'primary_classification': {
                'threat_type': 'unknown',
                'confidence': 0.1,
                'method': 'fallback'
            },
            'anomaly_detection': {
                'is_anomaly': False,
                'anomaly_score': 0.0
            },
            'rule_based': {
                'rules_triggered': [],
                'severity': 1,
                'is_threat': False
            },
            'final_classification': {
                'threat_type': 'unknown',
                'confidence': 0.1,
                'severity': 1,
                'is_threat': False,
                'classification_method': 'fallback'
            }
        }
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            # Save models
            if self.primary_classifier:
                joblib.dump(self.primary_classifier, self.model_dir / 'random_forest_classifier.pkl')
            
            if self.anomaly_detector:
                joblib.dump(self.anomaly_detector, self.model_dir / 'isolation_forest.pkl')
            
            if self.feature_selector:
                joblib.dump(self.feature_selector, self.model_dir / 'feature_selector.pkl')
            
            # Save preprocessing objects
            joblib.dump(self.scaler, self.model_dir / 'feature_scaler.pkl')
            joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.pkl')
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'threat_categories': self.threat_categories,
                'model_config': self.model_config,
                'metrics': asdict(self.metrics) if self.metrics else None,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.model_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Load models
            rf_path = self.model_dir / 'random_forest_classifier.pkl'
            if rf_path.exists():
                self.primary_classifier = joblib.load(rf_path)
                self.logger.info("Random Forest classifier loaded")
            
            iso_path = self.model_dir / 'isolation_forest.pkl'
            if iso_path.exists():
                self.anomaly_detector = joblib.load(iso_path)
                self.logger.info("Isolation Forest loaded")
            
            fs_path = self.model_dir / 'feature_selector.pkl'
            if fs_path.exists():
                self.feature_selector = joblib.load(fs_path)
                self.logger.info("Feature selector loaded")
            
            # Load preprocessing objects
            scaler_path = self.model_dir / 'feature_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            encoder_path = self.model_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            # Load metadata
            metadata_path = self.model_dir / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('metrics'):
                        self.metrics = ModelMetrics(**metadata['metrics'])
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        if not self.metrics:
            return
        
        report_path = self.model_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_path, 'w') as f:
            f.write("THREAT DETECTION MODEL TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Date: {self.metrics.last_trained}\n")
            f.write(f"Training Time: {self.metrics.training_time:.2f} seconds\n")
            f.write(f"Training Samples: {self.metrics.training_samples}\n")
            f.write(f"Test Samples: {self.metrics.test_samples}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Accuracy: {self.metrics.accuracy:.3f}\n")
            f.write(f"Precision: {self.metrics.precision:.3f}\n")
            f.write(f"Recall: {self.metrics.recall:.3f}\n")
            f.write(f"F1 Score: {self.metrics.f1_score:.3f}\n")
            f.write(f"AUC Score: {self.metrics.auc_score:.3f}\n\n")
            
            f.write("TOP FEATURE IMPORTANCE:\n")
            sorted_features = sorted(
                self.metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features[:10]:
                f.write(f"{feature}: {importance:.3f}\n")
            
            f.write(f"\nConfusion Matrix:\n{self.metrics.confusion_matrix}\n")
        
        self.logger.info(f"Training report saved to {report_path}")
    
    def evaluate_model(self, test_data: List[Dict], test_labels: List[str]) -> Dict[str, Any]:
        """Evaluate model on new test data"""
        if not self.primary_classifier:
            raise ValueError("Model not trained yet")
        
        # Extract features
        feature_vectors = []
        for event in test_data:
            features = self.extract_features(event)
            feature_vectors.append(features.to_vector())
        
        X_test = np.array(feature_vectors)
        y_test = self.label_encoder.transform(test_labels)
        
        # Apply preprocessing
        if self.feature_selector:
            X_test = self.feature_selector.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.primary_classifier.predict(X_test_scaled)
        y_pred_proba = self.primary_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'test_samples': len(test_data)
        }
    
    def retrain_with_new_data(self, new_events: List[Dict], new_labels: List[str]):
        """Retrain model with additional data"""
        self.logger.info(f"Retraining model with {len(new_events)} new samples...")
        
        # If no existing model, train from scratch
        if not self.primary_classifier:
            return self.train_model(new_events, new_labels)
        
        # Extract features from new data
        new_features = []
        for event in new_events:
            features = self.extract_features(event)
            new_features.append(features.to_vector())
        
        X_new = np.array(new_features)
        y_new = self.label_encoder.transform(new_labels)
        
        # Apply existing preprocessing
        if self.feature_selector:
            X_new = self.feature_selector.transform(X_new)
        X_new_scaled = self.scaler.transform(X_new)
        
        # Retrain with new data (incremental learning simulation)
        # Note: Random Forest doesn't support true incremental learning,
        # so we simulate it by training a new model with combined data
        
        # For now, we'll retrain the entire model
        # In production, you might want to implement online learning algorithms
        return self.train_model(new_events, new_labels)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_status': 'trained' if self.primary_classifier else 'not_trained',
            'feature_count': len(self.feature_names),
            'threat_categories': self.threat_categories,
            'model_config': self.model_config
        }
        
        if self.metrics:
            info['performance'] = asdict(self.metrics)
        
        if self.primary_classifier:
            info['model_details'] = {
                'n_estimators': getattr(self.primary_classifier, 'n_estimators', None),
                'max_depth': getattr(self.primary_classifier, 'max_depth', None),
                'n_features_in': getattr(self.primary_classifier, 'n_features_in_', None),
                'n_classes': getattr(self.primary_classifier, 'n_classes_', None)
            }
        
        return info

# Global model instance
threat_model = ThreatDetectionModel()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Threat Detection ML Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict threat for JSON event data')
    parser.add_argument('--info', action='store_true', help='Show model information')
    
    args = parser.parse_args()
    
    if args.train:
        print("🤖 Training threat detection model...")
        metrics = threat_model.train_model()
        print(f"✅ Training completed!")
        print(f"   Accuracy: {metrics.accuracy:.3f}")
        print(f"   F1 Score: {metrics.f1_score:.3f}")
        print(f"   Training time: {metrics.training_time:.2f}s")
    
    elif args.info:
        info = threat_model.get_model_info()
        print("🔍 Model Information:")
        print(json.dumps(info, indent=2, default=str))
    
    elif args.predict:
        try:
            event_data = json.loads(args.predict)
            result = threat_model.predict_threat(event_data)
            print("🎯 Threat Prediction:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("❌ Invalid JSON format for event data")
    
    else:
        parser.print_help()