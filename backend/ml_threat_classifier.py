#!/usr/bin/env python3
"""
ML-based Threat Classification System
Advanced machine learning models for threat detection and classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import re
import ipaddress

@dataclass
class ThreatFeatures:
    """Feature vector for threat classification"""
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

class ThreatClassifier:
    """Advanced ML-based threat classification system"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Threat categories
        self.threat_categories = [
            'benign', 'malware', 'botnet', 'ddos', 'brute_force',
            'sql_injection', 'xss', 'phishing', 'ransomware', 'apt'
        ]
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model performance metrics
        self.model_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.init_models()
    
    def init_models(self):
        """Initialize ML models"""
        try:
            # Load pre-trained models if available
            self.load_models()
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained models: {e}")
            self.logger.info("Initializing new models...")
            self.create_new_models()
    
    def create_new_models(self):
        """Create new ML models"""
        # Primary classifier (Random Forest)
        self.models['primary'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Anomaly detector (Isolation Forest)
        self.models['anomaly'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Clustering model (DBSCAN)
        self.models['clustering'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Feature scalers
        self.scalers['standard'] = StandardScaler()
        
        # Label encoders
        self.encoders['threat_type'] = LabelEncoder()
        self.encoders['threat_type'].fit(self.threat_categories)
        
        self.logger.info("New models initialized")
    
    def extract_features(self, event_data: Dict) -> ThreatFeatures:
        """Extract features from network event data"""
        # Network features
        packet_size = float(event_data.get('packet_size', 0))
        connection_duration = float(event_data.get('connection_duration', 0))
        bytes_transferred = float(event_data.get('bytes_transferred', 0))
        packets_per_second = float(event_data.get('packets_per_second', 0))
        unique_ports = int(event_data.get('unique_ports', 1))
        protocol_diversity = float(event_data.get('protocol_diversity', 0))
        
        # Behavioral features
        connection_frequency = float(event_data.get('connection_frequency', 0))
        time_between_connections = float(event_data.get('time_between_connections', 0))
        geographic_distance = float(event_data.get('geographic_distance', 0))
        reputation_score = float(event_data.get('reputation_score', 0.5))
        
        # Content analysis
        payload = event_data.get('payload', b'')
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
            country_risk_score=country_risk_score
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
            b'trojan', b'backdoor', b'malware', b'virus'
        ]
        
        count = 0
        payload_lower = payload.lower()
        
        for pattern in suspicious_patterns:
            count += payload_lower.count(pattern)
        
        return count
    
    def detect_base64(self, payload: bytes) -> bool:
        """Detect base64 encoded content"""
        try:
            # Look for base64 patterns
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
        return entropy > 7.0  # Threshold for encrypted data
    
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
    
    def classify_threat(self, event_data: Dict) -> Dict[str, Any]:
        """Classify threat using ML models"""
        try:
            # Extract features
            features = self.extract_features(event_data)
            feature_vector = self.features_to_vector(features)
            
            # Scale features
            if 'standard' in self.scalers and hasattr(self.scalers['standard'], 'mean_'):
                scaled_features = self.scalers['standard'].transform([feature_vector])
            else:
                scaled_features = [feature_vector]
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0',
                'features_used': len(feature_vector)
            }
            
            # Primary classification
            if 'primary' in self.models and hasattr(self.models['primary'], 'predict'):
                try:
                    prediction = self.models['primary'].predict(scaled_features)[0]
                    probabilities = self.models['primary'].predict_proba(scaled_features)[0]
                    
                    threat_type = self.encoders['threat_type'].inverse_transform([prediction])[0]
                    confidence = float(np.max(probabilities))
                    
                    results['primary_classification'] = {
                        'threat_type': threat_type,
                        'confidence': confidence,
                        'probabilities': {
                            category: float(prob) 
                            for category, prob in zip(self.threat_categories, probabilities)
                        }
                    }
                except Exception as e:
                    self.logger.warning(f"Primary classification failed: {e}")
                    results['primary_classification'] = self.fallback_classification(features)
            else:
                results['primary_classification'] = self.fallback_classification(features)
            
            # Anomaly detection
            if 'anomaly' in self.models and hasattr(self.models['anomaly'], 'predict'):
                try:
                    anomaly_score = self.models['anomaly'].decision_function(scaled_features)[0]
                    is_anomaly = self.models['anomaly'].predict(scaled_features)[0] == -1
                    
                    results['anomaly_detection'] = {
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': float(anomaly_score),
                        'threshold': 0.0
                    }
                except Exception as e:
                    self.logger.warning(f"Anomaly detection failed: {e}")
                    results['anomaly_detection'] = {
                        'is_anomaly': False,
                        'anomaly_score': 0.0,
                        'threshold': 0.0
                    }
            
            # Rule-based classification (fallback)
            rule_based = self.rule_based_classification(features)
            results['rule_based'] = rule_based
            
            # Combine results
            final_classification = self.combine_classifications(results)
            results['final_classification'] = final_classification
            
            return results
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return self.fallback_classification_dict(event_data)
    
    def features_to_vector(self, features: ThreatFeatures) -> List[float]:
        """Convert features to numerical vector"""
        return [
            features.packet_size,
            features.connection_duration,
            features.bytes_transferred,
            features.packets_per_second,
            float(features.unique_ports),
            features.protocol_diversity,
            features.connection_frequency,
            features.time_between_connections,
            features.geographic_distance,
            features.reputation_score,
            features.payload_entropy,
            float(features.suspicious_strings),
            float(features.base64_content),
            float(features.encrypted_content),
            float(features.hour_of_day),
            float(features.day_of_week),
            float(features.is_weekend),
            float(features.source_ip_class),
            float(features.is_tor_exit),
            float(features.is_vpn),
            features.country_risk_score
        ]
    
    def fallback_classification(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Rule-based fallback classification"""
        threat_type = 'benign'
        confidence = 0.5
        
        # High entropy suggests encryption/malware
        if features.payload_entropy > 7.5:
            threat_type = 'malware'
            confidence = 0.7
        
        # Many suspicious strings suggest attack
        elif features.suspicious_strings > 5:
            threat_type = 'malware'
            confidence = 0.8
        
        # High packet rate suggests DDoS
        elif features.packets_per_second > 1000:
            threat_type = 'ddos'
            confidence = 0.9
        
        # Low reputation score
        elif features.reputation_score < 0.3:
            threat_type = 'malware'
            confidence = 0.6
        
        return {
            'threat_type': threat_type,
            'confidence': confidence,
            'method': 'rule_based_fallback'
        }
    
    def rule_based_classification(self, features: ThreatFeatures) -> Dict[str, Any]:
        """Rule-based threat classification"""
        rules_triggered = []
        severity = 1
        
        # Network anomalies
        if features.packets_per_second > 500:
            rules_triggered.append('high_packet_rate')
            severity = max(severity, 7)
        
        if features.bytes_transferred > 100_000_000:  # 100MB
            rules_triggered.append('large_data_transfer')
            severity = max(severity, 6)
        
        # Content analysis
        if features.suspicious_strings > 3:
            rules_triggered.append('suspicious_content')
            severity = max(severity, 8)
        
        if features.payload_entropy > 7.0:
            rules_triggered.append('high_entropy_payload')
            severity = max(severity, 6)
        
        # Behavioral analysis
        if features.connection_frequency > 100:
            rules_triggered.append('rapid_connections')
            severity = max(severity, 7)
        
        # Source reputation
        if features.reputation_score < 0.2:
            rules_triggered.append('poor_reputation')
            severity = max(severity, 8)
        
        if features.is_tor_exit:
            rules_triggered.append('tor_exit_node')
            severity = max(severity, 5)
        
        # Geographic anomalies
        if features.geographic_distance > 10000:  # 10,000 km
            rules_triggered.append('geographic_anomaly')
            severity = max(severity, 4)
        
        return {
            'rules_triggered': rules_triggered,
            'severity': severity,
            'is_threat': len(rules_triggered) > 0
        }
    
    def combine_classifications(self, results: Dict) -> Dict[str, Any]:
        """Combine multiple classification results"""
        primary = results.get('primary_classification', {})
        anomaly = results.get('anomaly_detection', {})
        rule_based = results.get('rule_based', {})
        
        # Start with primary classification
        threat_type = primary.get('threat_type', 'benign')
        confidence = primary.get('confidence', 0.5)
        severity = rule_based.get('severity', 1)
        
        # Adjust based on anomaly detection
        if anomaly.get('is_anomaly', False):
            confidence = min(1.0, confidence + 0.2)
            severity = max(severity, 6)
        
        # Adjust based on rule-based results
        if rule_based.get('is_threat', False):
            if threat_type == 'benign':
                threat_type = 'malware'  # Generic threat type
            confidence = min(1.0, confidence + 0.1)
            severity = max(severity, rule_based.get('severity', 1))
        
        return {
            'threat_type': threat_type,
            'confidence': float(confidence),
            'severity': int(severity),
            'is_threat': threat_type != 'benign' or rule_based.get('is_threat', False),
            'classification_method': 'combined'
        }
    
    def fallback_classification_dict(self, event_data: Dict) -> Dict[str, Any]:
        """Complete fallback classification"""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0',
            'features_used': 0,
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
    
    def train_model(self, training_data: List[Dict], labels: List[str]):
        """Train the ML models with new data"""
        try:
            # Extract features from training data
            feature_vectors = []
            for event in training_data:
                features = self.extract_features(event)
                feature_vectors.append(self.features_to_vector(features))
            
            X = np.array(feature_vectors)
            y = np.array(labels)
            
            # Encode labels
            y_encoded = self.encoders['threat_type'].transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Train primary classifier
            self.models['primary'].fit(X_train_scaled, y_train)
            
            # Train anomaly detector (unsupervised)
            self.models['anomaly'].fit(X_train_scaled)
            
            # Evaluate model
            y_pred = self.models['primary'].predict(X_test_scaled)
            
            # Store metrics
            self.model_metrics['accuracy'] = float(np.mean(y_pred == y_test))
            self.model_metrics['training_samples'] = len(X_train)
            self.model_metrics['last_trained'] = datetime.now().isoformat()
            
            # Feature importance
            if hasattr(self.models['primary'], 'feature_importances_'):
                self.feature_importance = {
                    f'feature_{i}': float(importance)
                    for i, importance in enumerate(self.models['primary'].feature_importances_)
                }
            
            self.logger.info(f"Model trained successfully. Accuracy: {self.model_metrics['accuracy']:.3f}")
            
            # Save models
            self.save_models()
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                joblib.dump(model, f"{self.model_path}/{name}_model.pkl")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f"{self.model_path}/{name}_scaler.pkl")
            
            # Save encoders
            for name, encoder in self.encoders.items():
                joblib.dump(encoder, f"{self.model_path}/{name}_encoder.pkl")
            
            # Save metadata
            metadata = {
                'model_metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'threat_categories': self.threat_categories,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        import os
        
        # Load models
        for name in ['primary', 'anomaly', 'clustering']:
            model_file = f"{self.model_path}/{name}_model.pkl"
            if os.path.exists(model_file):
                self.models[name] = joblib.load(model_file)
        
        # Load scalers
        for name in ['standard']:
            scaler_file = f"{self.model_path}/{name}_scaler.pkl"
            if os.path.exists(scaler_file):
                self.scalers[name] = joblib.load(scaler_file)
        
        # Load encoders
        for name in ['threat_type']:
            encoder_file = f"{self.model_path}/{name}_encoder.pkl"
            if os.path.exists(encoder_file):
                self.encoders[name] = joblib.load(encoder_file)
        
        # Load metadata
        metadata_file = f"{self.model_path}/metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.model_metrics = metadata.get('model_metrics', {})
                self.feature_importance = metadata.get('feature_importance', {})
        
        self.logger.info("Models loaded successfully")

# Global classifier instance
threat_classifier = ThreatClassifier()

if __name__ == "__main__":
    # Test the classifier
    test_event = {
        'packet_size': 1024,
        'connection_duration': 5.0,
        'bytes_transferred': 10000,
        'packets_per_second': 100,
        'payload': b'GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n',
        'source_ip': '192.168.1.100',
        'reputation_score': 0.8,
        'timestamp': datetime.now().isoformat()
    }
    
    result = threat_classifier.classify_threat(test_event)
    print(json.dumps(result, indent=2))