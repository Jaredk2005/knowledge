#!/usr/bin/env python3
"""
Critical Incidents Detection and Management Backend
Real-time detection, classification, and notification system for critical security incidents
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

# Web framework and async support
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests

# Database
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

@dataclass
class CriticalIncident:
    """Critical incident data structure"""
    id: str
    incident_number: str
    title: str
    description: str
    incident_type: str
    severity: str
    status: str
    source_ip: Optional[str]
    destination_ip: Optional[str]
    source_system: Optional[str]
    target_system: Optional[str]
    affected_systems: List[str]
    indicators_of_compromise: List[str]
    mitre_tactics: List[str]
    confidence_score: int
    risk_score: int
    detected_at: datetime
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    raw_data: Dict[str, Any]

class CriticalIncidentDetector:
    """Critical incident detection engine"""
    
    def __init__(self):
        self.active_incidents = {}
        self.incident_history = deque(maxlen=1000)
        self.detection_rules = self.load_detection_rules()
        self.correlation_window = 300  # 5 minutes
        self.critical_threshold = 8  # Severity threshold for critical incidents
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_detection_rules(self) -> Dict[str, Dict]:
        """Load critical incident detection rules"""
        return {
            'multiple_failed_logins': {
                'threshold': 10,
                'timeframe': 300,  # 5 minutes
                'severity': 'critical',
                'description': 'Multiple failed login attempts detected',
                'mitre_tactics': ['Initial Access', 'Credential Access']
            },
            'ransomware_indicators': {
                'patterns': ['encrypt', 'ransom', 'bitcoin', '.locked', '.encrypted'],
                'severity': 'critical',
                'description': 'Ransomware activity detected',
                'mitre_tactics': ['Impact', 'Defense Evasion']
            },
            'data_exfiltration': {
                'threshold': 100_000_000,  # 100MB
                'timeframe': 600,  # 10 minutes
                'severity': 'critical',
                'description': 'Large data transfer detected - possible exfiltration',
                'mitre_tactics': ['Exfiltration', 'Collection']
            },
            'privilege_escalation': {
                'patterns': ['sudo', 'admin', 'root', 'administrator'],
                'severity': 'critical',
                'description': 'Privilege escalation attempt detected',
                'mitre_tactics': ['Privilege Escalation', 'Defense Evasion']
            },
            'lateral_movement': {
                'threshold': 5,  # connections to different systems
                'timeframe': 300,
                'severity': 'high',
                'description': 'Lateral movement detected across network',
                'mitre_tactics': ['Lateral Movement', 'Discovery']
            },
            'c2_communication': {
                'patterns': ['bot', 'command', 'control', 'beacon'],
                'severity': 'critical',
                'description': 'Command and control communication detected',
                'mitre_tactics': ['Command and Control', 'Persistence']
            }
        }
    
    async def analyze_event(self, event_data: Dict) -> Optional[CriticalIncident]:
        """Analyze event for critical incident patterns"""
        incidents_detected = []
        
        # Check each detection rule
        for rule_name, rule_config in self.detection_rules.items():
            if await self.check_rule(rule_name, rule_config, event_data):
                incident = await self.create_incident(rule_name, rule_config, event_data)
                if incident:
                    incidents_detected.append(incident)
        
        # Return the highest severity incident
        if incidents_detected:
            critical_incidents = [i for i in incidents_detected if i.severity == 'critical']
            return critical_incidents[0] if critical_incidents else incidents_detected[0]
        
        return None
    
    async def check_rule(self, rule_name: str, rule_config: Dict, event_data: Dict) -> bool:
        """Check if event matches detection rule"""
        try:
            if rule_name == 'multiple_failed_logins':
                return await self.check_failed_logins(rule_config, event_data)
            elif rule_name == 'ransomware_indicators':
                return self.check_patterns(rule_config['patterns'], event_data)
            elif rule_name == 'data_exfiltration':
                return await self.check_data_transfer(rule_config, event_data)
            elif rule_name == 'privilege_escalation':
                return self.check_patterns(rule_config['patterns'], event_data)
            elif rule_name == 'lateral_movement':
                return await self.check_lateral_movement(rule_config, event_data)
            elif rule_name == 'c2_communication':
                return self.check_patterns(rule_config['patterns'], event_data)
        except Exception as e:
            self.logger.error(f"Error checking rule {rule_name}: {e}")
        
        return False
    
    async def check_failed_logins(self, rule_config: Dict, event_data: Dict) -> bool:
        """Check for multiple failed login attempts"""
        if 'failed' not in event_data.get('description', '').lower():
            return False
        
        source_ip = event_data.get('source_ip')
        if not source_ip:
            return False
        
        # Count recent failed logins from same IP
        now = datetime.now()
        timeframe = timedelta(seconds=rule_config['timeframe'])
        
        try:
            # Query recent incidents from database
            response = supabase.table('incidents').select('*').gte(
                'created_at', (now - timeframe).isoformat()
            ).eq('source_ip', source_ip).ilike('description', '%failed%').execute()
            
            failed_count = len(response.data) if response.data else 0
            return failed_count >= rule_config['threshold']
        except Exception as e:
            self.logger.error(f"Error checking failed logins: {e}")
            return False
    
    def check_patterns(self, patterns: List[str], event_data: Dict) -> bool:
        """Check if event contains suspicious patterns"""
        text_fields = [
            event_data.get('description', ''),
            event_data.get('title', ''),
            str(event_data.get('raw_data', {}))
        ]
        
        combined_text = ' '.join(text_fields).lower()
        
        return any(pattern.lower() in combined_text for pattern in patterns)
    
    async def check_data_transfer(self, rule_config: Dict, event_data: Dict) -> bool:
        """Check for large data transfers"""
        bytes_transferred = event_data.get('bytes_transferred', 0)
        if bytes_transferred < rule_config['threshold']:
            return False
        
        # Additional checks for suspicious destinations
        dest_ip = event_data.get('destination_ip')
        if dest_ip and self.is_external_ip(dest_ip):
            return True
        
        return False
    
    async def check_lateral_movement(self, rule_config: Dict, event_data: Dict) -> bool:
        """Check for lateral movement patterns"""
        source_ip = event_data.get('source_ip')
        if not source_ip:
            return False
        
        now = datetime.now()
        timeframe = timedelta(seconds=rule_config['timeframe'])
        
        try:
            # Count unique destination systems from same source
            response = supabase.table('network_traffic').select('destination_ip').gte(
                'timestamp', (now - timeframe).isoformat()
            ).eq('source_ip', source_ip).execute()
            
            if response.data:
                unique_destinations = len(set(row['destination_ip'] for row in response.data))
                return unique_destinations >= rule_config['threshold']
        except Exception as e:
            self.logger.error(f"Error checking lateral movement: {e}")
        
        return False
    
    def is_external_ip(self, ip: str) -> bool:
        """Check if IP is external (not private)"""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return not ip_obj.is_private
        except:
            return False
    
    async def create_incident(self, rule_name: str, rule_config: Dict, event_data: Dict) -> Optional[CriticalIncident]:
        """Create critical incident from detected event"""
        try:
            incident_id = str(uuid.uuid4())
            incident_number = f"CRIT-{datetime.now().strftime('%Y%m%d')}-{incident_id[:8]}"
            
            incident = CriticalIncident(
                id=incident_id,
                incident_number=incident_number,
                title=f"Critical Incident: {rule_config['description']}",
                description=rule_config['description'],
                incident_type=rule_name.replace('_', ' '),
                severity=rule_config['severity'],
                status='detected',
                source_ip=event_data.get('source_ip'),
                destination_ip=event_data.get('destination_ip'),
                source_system=event_data.get('source_system'),
                target_system=event_data.get('target_system'),
                affected_systems=event_data.get('affected_systems', []),
                indicators_of_compromise=self.extract_iocs(event_data),
                mitre_tactics=rule_config.get('mitre_tactics', []),
                confidence_score=85,  # High confidence for rule-based detection
                risk_score=90 if rule_config['severity'] == 'critical' else 70,
                detected_at=datetime.now(),
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                raw_data=event_data
            )
            
            # Store in database
            await self.store_incident(incident)
            
            # Store in memory
            self.active_incidents[incident.id] = incident
            self.incident_history.append(incident)
            
            self.logger.critical(f"CRITICAL INCIDENT DETECTED: {incident.incident_number} - {incident.description}")
            
            return incident
            
        except Exception as e:
            self.logger.error(f"Error creating incident: {e}")
            return None
    
    def extract_iocs(self, event_data: Dict) -> List[str]:
        """Extract indicators of compromise from event data"""
        iocs = []
        
        if event_data.get('source_ip'):
            iocs.append(f"Source IP: {event_data['source_ip']}")
        
        if event_data.get('destination_ip'):
            iocs.append(f"Destination IP: {event_data['destination_ip']}")
        
        if event_data.get('file_hash'):
            iocs.append(f"File Hash: {event_data['file_hash']}")
        
        if event_data.get('domain'):
            iocs.append(f"Domain: {event_data['domain']}")
        
        return iocs
    
    async def store_incident(self, incident: CriticalIncident):
        """Store incident in database"""
        try:
            incident_data = {
                'id': incident.id,
                'incident_number': incident.incident_number,
                'title': incident.title,
                'description': incident.description,
                'incident_type': incident.incident_type,
                'severity': incident.severity,
                'status': incident.status,
                'source_ip': incident.source_ip,
                'destination_ip': incident.destination_ip,
                'source_system': incident.source_system,
                'target_system': incident.target_system,
                'affected_systems': incident.affected_systems,
                'indicators_of_compromise': incident.indicators_of_compromise,
                'mitre_tactics': incident.mitre_tactics,
                'confidence_score': incident.confidence_score,
                'risk_score': incident.risk_score,
                'detected_at': incident.detected_at.isoformat(),
                'first_seen': incident.first_seen.isoformat() if incident.first_seen else None,
                'last_seen': incident.last_seen.isoformat() if incident.last_seen else None
            }
            
            response = supabase.table('incidents').insert(incident_data).execute()
            
            if response.data:
                self.logger.info(f"Incident stored in database: {incident.incident_number}")
            else:
                self.logger.error(f"Failed to store incident: {response}")
                
        except Exception as e:
            self.logger.error(f"Error storing incident in database: {e}")
    
    def get_critical_incidents(self, limit: int = 50) -> List[Dict]:
        """Get recent critical incidents"""
        try:
            response = supabase.table('incidents').select('*').eq(
                'severity', 'critical'
            ).order('detected_at', desc=True).limit(limit).execute()
            
            return response.data if response.data else []
        except Exception as e:
            self.logger.error(f"Error fetching critical incidents: {e}")
            return []
    
    def get_incident_statistics(self) -> Dict:
        """Get critical incident statistics"""
        try:
            now = datetime.now()
            hour_ago = (now - timedelta(hours=1)).isoformat()
            day_ago = (now - timedelta(days=1)).isoformat()
            
            # Get critical incidents from last 24 hours
            response = supabase.table('incidents').select('*').eq(
                'severity', 'critical'
            ).gte('detected_at', day_ago).execute()
            
            incidents = response.data if response.data else []
            
            # Calculate statistics
            total_critical = len(incidents)
            last_hour = len([i for i in incidents if i['detected_at'] >= hour_ago])
            active = len([i for i in incidents if i['status'] in ['detected', 'investigating']])
            
            # Group by type
            incident_types = defaultdict(int)
            for incident in incidents:
                incident_types[incident['incident_type']] += 1
            
            return {
                'total_critical': total_critical,
                'critical_last_hour': last_hour,
                'active_critical': active,
                'incident_types': dict(incident_types),
                'avg_risk_score': sum(i.get('risk_score', 0) for i in incidents) / max(len(incidents), 1)
            }
        except Exception as e:
            self.logger.error(f"Error getting incident statistics: {e}")
            return {
                'total_critical': 0,
                'critical_last_hour': 0,
                'active_critical': 0,
                'incident_types': {},
                'avg_risk_score': 0
            }

# Flask Web API
app = Flask(__name__)
app.config['SECRET_KEY'] = 'critical_incidents_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global detector instance
incident_detector = CriticalIncidentDetector()

@app.route('/api/critical-incidents', methods=['GET'])
def get_critical_incidents():
    """Get critical incidents"""
    limit = request.args.get('limit', 50, type=int)
    incidents = incident_detector.get_critical_incidents(limit)
    return jsonify({'incidents': incidents, 'total': len(incidents)})

@app.route('/api/critical-incidents/statistics', methods=['GET'])
def get_critical_statistics():
    """Get critical incident statistics"""
    stats = incident_detector.get_incident_statistics()
    return jsonify(stats)

@app.route('/api/critical-incidents/analyze', methods=['POST'])
def analyze_event():
    """Analyze event for critical incidents"""
    event_data = request.json
    
    async def process_event():
        incident = await incident_detector.analyze_event(event_data)
        if incident:
            # Emit real-time notification
            socketio.emit('critical_incident_detected', {
                'id': incident.id,
                'incident_number': incident.incident_number,
                'title': incident.title,
                'description': incident.description,
                'severity': incident.severity,
                'detected_at': incident.detected_at.isoformat(),
                'source_ip': incident.source_ip,
                'risk_score': incident.risk_score
            })
            
            return asdict(incident)
        return None
    
    # Run async processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_event())
        if result:
            return jsonify({'status': 'critical_incident_detected', 'incident': result})
        else:
            return jsonify({'status': 'no_critical_incident'})
    finally:
        loop.close()

@app.route('/api/critical-incidents/<incident_id>/resolve', methods=['POST'])
def resolve_incident(incident_id):
    """Resolve a critical incident"""
    try:
        response = supabase.table('incidents').update({
            'status': 'resolved',
            'resolved_at': datetime.now().isoformat()
        }).eq('id', incident_id).execute()
        
        if response.data:
            # Emit real-time update
            socketio.emit('incident_resolved', {
                'incident_id': incident_id,
                'status': 'resolved',
                'resolved_at': datetime.now().isoformat()
            })
            
            return jsonify({'status': 'success', 'message': 'Incident resolved'})
        else:
            return jsonify({'status': 'error', 'message': 'Incident not found'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_incidents': len(incident_detector.active_incidents)
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('connected', {'status': 'Connected to critical incidents system'})

@socketio.on('subscribe_critical_incidents')
def handle_subscribe():
    """Subscribe to critical incident updates"""
    emit('subscribed', {'message': 'Subscribed to critical incident updates'})

def simulate_critical_events():
    """Simulate critical events for testing"""
    import random
    
    sample_events = [
        {
            'description': 'Multiple failed login attempts from suspicious IP',
            'source_ip': '192.168.1.100',
            'source_system': 'web-server-01',
            'event_type': 'authentication_failure'
        },
        {
            'description': 'Ransomware encryption activity detected',
            'source_ip': '10.0.0.50',
            'target_system': 'file-server',
            'file_hash': 'a1b2c3d4e5f6789',
            'event_type': 'file_encryption'
        },
        {
            'description': 'Large data transfer to external IP',
            'source_ip': '172.16.0.25',
            'destination_ip': '203.0.113.45',
            'bytes_transferred': 150_000_000,
            'event_type': 'data_transfer'
        }
    ]
    
    while True:
        if random.random() < 0.1:  # 10% chance every cycle
            event = random.choice(sample_events).copy()
            event['timestamp'] = datetime.now().isoformat()
            
            async def process_sim_event():
                await incident_detector.analyze_event(event)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(process_sim_event())
            finally:
                loop.close()
        
        time.sleep(random.uniform(10, 30))  # Wait 10-30 seconds

if __name__ == '__main__':
    print("🚨 Critical Incidents Detection Backend")
    print("=" * 40)
    print("🔍 Real-time critical incident detection")
    print("📊 Advanced correlation and analysis")
    print("🚨 Automated alerting and response")
    print("🌐 REST API and WebSocket support")
    print()
    print("API Endpoints:")
    print("  GET  /api/critical-incidents - Get critical incidents")
    print("  GET  /api/critical-incidents/statistics - Get statistics")
    print("  POST /api/critical-incidents/analyze - Analyze event")
    print("  POST /api/critical-incidents/<id>/resolve - Resolve incident")
    print("  GET  /api/health - Health check")
    print()
    
    # Start simulation thread for testing
    simulation_thread = threading.Thread(target=simulate_critical_events, daemon=True)
    simulation_thread.start()
    
    # Start the web server
    socketio.run(app, host='0.0.0.0', port=8002, debug=False)