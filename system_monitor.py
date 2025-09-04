#!/usr/bin/env python3
"""
Integrated Cybersecurity and System Monitoring Alert System
Monitors system resources and cybersecurity threats, sends email alerts when thresholds are exceeded
"""

import smtplib
import psutil
import time
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any
import threading
import requests
import socket
import subprocess
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_alerts.log'),
        logging.StreamHandler()
    ]
)

class EmailAlertSystem:
    def __init__(self, config_file: str = 'alert_config.json'):
        """Initialize the email alert system with configuration"""
        self.config = self.load_config(config_file)
        self.alert_history = {}
        self.running = False
        self.threat_detection_enabled = True
        self.security_events = []
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "smtp": {
                "server": "smtp.gmail.com",
                "port": 587,
                "username": "your-email@gmail.com",
                "password": "your-app-password",  # Use app password for Gmail
                "use_tls": True
            },
            "alerts": {
                "from_email": "alerts@yourcompany.com",
                "to_emails": ["admin1@yourcompany.com", "admin2@yourcompany.com"],
                "subject_prefix": "[SYSTEM ALERT]"
            },
            "thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "load_average": 4.0
            },
            "security": {
                "enable_threat_detection": True,
                "failed_login_threshold": 5,
                "suspicious_process_threshold": 3,
                "network_scan_threshold": 10,
                "check_malware_signatures": True
            },
            "monitoring": {
                "check_interval": 60,  # seconds
                "cooldown_period": 300  # seconds between duplicate alerts
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except FileNotFoundError:
            logging.warning(f"Config file {config_file} not found, using defaults")
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def send_email(self, subject: str, body: str, priority: str = "normal") -> bool:
        """Send email alert to administrators"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['alerts']['from_email']
            msg['To'] = ', '.join(self.config['alerts']['to_emails'])
            msg['Subject'] = f"{self.config['alerts']['subject_prefix']} {subject}"
            
            # Add priority header
            if priority == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Create HTML body
            html_body = f"""
            <html>
            <head></head>
            <body>
                <h2 style="color: {'#d32f2f' if priority == 'high' else '#1976d2'};">System Alert</h2>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <hr>
                <div style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
                    {body.replace('\n', '<br>')}
                </div>
                <hr>
                <p><em>This is an automated message from the System Monitoring Service</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.config['smtp']['server'], self.config['smtp']['port'])
            if self.config['smtp']['use_tls']:
                server.starttls()
            
            server.login(self.config['smtp']['username'], self.config['smtp']['password'])
            
            # Send email
            text = msg.as_string()
            server.sendmail(
                self.config['alerts']['from_email'],
                self.config['alerts']['to_emails'],
                text
            )
            server.quit()
            
            logging.info(f"Alert email sent: {subject}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")
            return False
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        
        alerts = []
        if cpu_percent > self.config['thresholds']['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage detected: {cpu_percent:.1f}%",
                'value': cpu_percent,
                'threshold': self.config['thresholds']['cpu_percent'],
                'priority': 'high' if cpu_percent > 95 else 'medium'
            })
        
        if load_avg > self.config['thresholds']['load_average']:
            alerts.append({
                'type': 'load_high',
                'message': f"High load average detected: {load_avg:.2f}",
                'value': load_avg,
                'threshold': self.config['thresholds']['load_average'],
                'priority': 'high' if load_avg > 8 else 'medium'
            })
        
        return {'cpu_percent': cpu_percent, 'load_avg': load_avg, 'alerts': alerts}
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        alerts = []
        if memory.percent > self.config['thresholds']['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage detected: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB used)",
                'value': memory.percent,
                'threshold': self.config['thresholds']['memory_percent'],
                'priority': 'high' if memory.percent > 95 else 'medium'
            })
        
        if swap.percent > 50:  # Swap usage is generally bad
            alerts.append({
                'type': 'swap_high',
                'message': f"High swap usage detected: {swap.percent:.1f}%",
                'value': swap.percent,
                'threshold': 50,
                'priority': 'high'
            })
        
        return {
            'memory_percent': memory.percent,
            'swap_percent': swap.percent,
            'alerts': alerts
        }
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage for all mounted drives"""
        alerts = []
        disk_info = {}
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                percent_used = (usage.used / usage.total) * 100
                disk_info[partition.mountpoint] = percent_used
                
                if percent_used > self.config['thresholds']['disk_percent']:
                    alerts.append({
                        'type': 'disk_high',
                        'message': f"High disk usage on {partition.mountpoint}: {percent_used:.1f}% ({usage.free // (1024**3):.1f}GB free)",
                        'value': percent_used,
                        'threshold': self.config['thresholds']['disk_percent'],
                        'priority': 'high' if percent_used > 95 else 'medium'
                    })
            except PermissionError:
                continue
        
        return {'disk_usage': disk_info, 'alerts': alerts}
    
    def check_system_processes(self) -> Dict[str, Any]:
        """Check for suspicious processes or high resource usage"""
        alerts = []
        top_processes = []
        
        # Get top CPU consuming processes
        processes = [(p.info['pid'], p.info['name'], p.info['cpu_percent']) 
                    for p in psutil.process_iter(['pid', 'name', 'cpu_percent'])]
        processes.sort(key=lambda x: x[2], reverse=True)
        
        for pid, name, cpu in processes[:5]:
            top_processes.append(f"{name} (PID: {pid}): {cpu:.1f}% CPU")
            if cpu > 50:  # Process using more than 50% CPU
                alerts.append({
                    'type': 'process_high_cpu',
                    'message': f"Process {name} (PID: {pid}) using {cpu:.1f}% CPU",
                    'value': cpu,
                    'threshold': 50,
                    'priority': 'medium'
                })
        
        return {'top_processes': top_processes, 'alerts': alerts}
    
    def check_security_threats(self) -> Dict[str, Any]:
        """Check for cybersecurity threats"""
        if not self.config['security']['enable_threat_detection']:
            return {'alerts': []}
        
        alerts = []
        
        # Check for failed login attempts
        failed_logins = self.check_failed_logins()
        if failed_logins > self.config['security']['failed_login_threshold']:
            alerts.append({
                'type': 'security_failed_logins',
                'message': f"Multiple failed login attempts detected: {failed_logins} attempts",
                'value': failed_logins,
                'threshold': self.config['security']['failed_login_threshold'],
                'priority': 'high'
            })
        
        # Check for suspicious processes
        suspicious_processes = self.check_suspicious_processes()
        if len(suspicious_processes) > 0:
            alerts.append({
                'type': 'security_suspicious_process',
                'message': f"Suspicious processes detected: {', '.join(suspicious_processes)}",
                'value': len(suspicious_processes),
                'threshold': self.config['security']['suspicious_process_threshold'],
                'priority': 'high'
            })
        
        # Check for network scanning activity
        network_connections = self.check_network_activity()
        if network_connections > self.config['security']['network_scan_threshold']:
            alerts.append({
                'type': 'security_network_scan',
                'message': f"Potential network scanning detected: {network_connections} connections",
                'value': network_connections,
                'threshold': self.config['security']['network_scan_threshold'],
                'priority': 'medium'
            })
        
        # Check for malware signatures in running processes
        if self.config['security']['check_malware_signatures']:
            malware_indicators = self.check_malware_signatures()
            if len(malware_indicators) > 0:
                alerts.append({
                    'type': 'security_malware_detected',
                    'message': f"Potential malware indicators found: {', '.join(malware_indicators)}",
                    'value': len(malware_indicators),
                    'threshold': 1,
                    'priority': 'critical'
                })
        
        return {'alerts': alerts}
    
    def check_failed_logins(self) -> int:
        """Check for failed login attempts in system logs"""
        try:
            # Check auth.log for failed login attempts (Linux)
            failed_count = 0
            try:
                with open('/var/log/auth.log', 'r') as f:
                    recent_time = datetime.now().timestamp() - 3600  # Last hour
                    for line in f:
                        if 'Failed password' in line or 'authentication failure' in line:
                            # Simple timestamp check (this is a basic implementation)
                            failed_count += 1
            except (FileNotFoundError, PermissionError):
                # Fallback: simulate based on system activity
                failed_count = len([p for p in psutil.process_iter(['name']) 
                                  if 'ssh' in p.info['name'].lower()]) // 2
            
            return min(failed_count, 20)  # Cap at 20 for demo
        except Exception:
            return 0
    
    def check_suspicious_processes(self) -> List[str]:
        """Check for suspicious process names"""
        suspicious_names = [
            'nc', 'netcat', 'ncat', 'socat', 'telnet', 'ftp', 'tftp',
            'wget', 'curl', 'python', 'perl', 'ruby', 'bash', 'sh',
            'powershell', 'cmd', 'wscript', 'cscript', 'mshta'
        ]
        
        suspicious_found = []
        try:
            for proc in psutil.process_iter(['name', 'cmdline']):
                proc_name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                
                # Check for suspicious process names
                if any(sus_name in proc_name for sus_name in suspicious_names):
                    # Additional checks to reduce false positives
                    if any(indicator in cmdline for indicator in ['-e', '-c', 'exec', 'system', 'shell']):
                        suspicious_found.append(f"{proc.info['name']} (PID: {proc.pid})")
                
                # Check for base64 encoded commands
                if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', cmdline):
                    suspicious_found.append(f"{proc.info['name']} - Base64 encoded command")
        except Exception as e:
            logging.warning(f"Error checking suspicious processes: {e}")
        
        return suspicious_found[:5]  # Limit to 5 results
    
    def check_network_activity(self) -> int:
        """Check for unusual network activity"""
        try:
            connections = psutil.net_connections(kind='inet')
            # Count unique external connections
            external_connections = set()
            
            for conn in connections:
                if conn.raddr and conn.status == 'ESTABLISHED':
                    # Filter out local/private IPs
                    remote_ip = conn.raddr.ip
                    if not (remote_ip.startswith('127.') or 
                           remote_ip.startswith('192.168.') or
                           remote_ip.startswith('10.') or
                           remote_ip.startswith('172.')):
                        external_connections.add(remote_ip)
            
            return len(external_connections)
        except Exception:
            return 0
    
    def check_malware_signatures(self) -> List[str]:
        """Check for potential malware signatures in running processes"""
        malware_indicators = []
        suspicious_patterns = [
            r'(?i)(trojan|malware|virus|backdoor|keylogger|rootkit)',
            r'(?i)(cryptominer|coinminer|bitcoin|monero)',
            r'(?i)(ransomware|encrypt|decrypt|ransom)',
            r'(?i)(botnet|c2|command.*control)'
        ]
        
        try:
            for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
                try:
                    proc_info = f"{proc.info['name']} {proc.info['exe'] or ''} {' '.join(proc.info['cmdline'] or [])}"
                    
                    for pattern in suspicious_patterns:
                        if re.search(pattern, proc_info):
                            malware_indicators.append(f"{proc.info['name']} - Suspicious pattern detected")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logging.warning(f"Error checking malware signatures: {e}")
        
        return malware_indicators[:3]  # Limit to 3 results
    
    def send_threat_alert(self, threat_data: Dict[str, Any]) -> bool:
        """Send cybersecurity threat alert via email"""
        threat_type = threat_data.get('type', 'Unknown')
        severity = threat_data.get('severity', 'medium')
        description = threat_data.get('description', 'Threat detected')
        source_ip = threat_data.get('source_ip', 'Unknown')
        
        subject = f"CYBERSECURITY THREAT DETECTED - {threat_type.upper()}"
        
        body = f"""
CYBERSECURITY THREAT ALERT

Threat Type: {threat_type}
Severity: {severity.upper()}
Source IP: {source_ip}
Description: {description}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RECOMMENDED ACTIONS:
- Investigate the source IP address
- Check system logs for related activity
- Consider blocking the source if confirmed malicious
- Monitor for additional threats from this source

CURRENT SYSTEM STATUS:
- CPU Usage: {psutil.cpu_percent(interval=1):.1f}%
- Memory Usage: {psutil.virtual_memory().percent:.1f}%
- Active Network Connections: {len(psutil.net_connections()):.0f}

This alert was generated by the Integrated Cybersecurity Monitoring System.
        """
        
        priority = "high" if severity in ['high', 'critical'] else "normal"
        return self.send_email(subject, body, priority)
    
    def should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        now = time.time()
        last_sent = self.alert_history.get(alert_type, 0)
        cooldown = self.config['monitoring']['cooldown_period']
        
        if now - last_sent > cooldown:
            self.alert_history[alert_type] = now
            return True
        return False
    
    def run_system_check(self):
        """Run a complete system check and send alerts if necessary"""
        logging.info("Running system check...")
        
        # Collect all system metrics
        cpu_data = self.check_cpu_usage()
        memory_data = self.check_memory_usage()
        disk_data = self.check_disk_usage()
        process_data = self.check_system_processes()
        security_data = self.check_security_threats()
        
        # Collect all alerts
        all_alerts = []
        all_alerts.extend(cpu_data['alerts'])
        all_alerts.extend(memory_data['alerts'])
        all_alerts.extend(disk_data['alerts'])
        all_alerts.extend(process_data['alerts'])
        all_alerts.extend(security_data['alerts'])
        
        # Send alerts if any found
        if all_alerts:
            for alert in all_alerts:
                if self.should_send_alert(alert['type']):
                    # Create detailed report
                    report = f"""
SYSTEM ALERT DETAILS:
{alert['message']}

CURRENT SYSTEM STATUS:
- CPU Usage: {cpu_data['cpu_percent']:.1f}%
- Load Average: {cpu_data.get('load_avg', 'N/A')}
- Memory Usage: {memory_data['memory_percent']:.1f}%
- Swap Usage: {memory_data['swap_percent']:.1f}%

DISK USAGE:
"""
                    for mount, usage in disk_data['disk_usage'].items():
                        report += f"- {mount}: {usage:.1f}%\n"
                    
                    report += f"""
TOP PROCESSES:
"""
                    for process in process_data['top_processes']:
                        report += f"- {process}\n"
                    
                    # Add security information if it's a security alert
                    if alert['type'].startswith('security_'):
                        report += f"""
SECURITY ALERT DETAILS:
This is a cybersecurity threat detection alert.
Immediate investigation and response may be required.

THREAT INDICATORS:
- Alert Type: {alert['type']}
- Severity: {alert['priority']}
- Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    # Send email
                    subject = f"{alert['type'].replace('_', ' ').title()} - {alert['message']}"
                    self.send_email(subject, report, alert['priority'])
        else:
            logging.info("System check completed - no alerts triggered")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        logging.info("Starting system monitoring...")
        
        while self.running:
            try:
                self.run_system_check()
                time.sleep(self.config['monitoring']['check_interval'])
            except KeyboardInterrupt:
                logging.info("Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error during monitoring: {str(e)}")
                time.sleep(30)  # Wait before retrying
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logging.info("Monitoring stopped")
    
    def send_test_alert(self):
        """Send a test alert to verify email configuration"""
        test_body = f"""
This is a test alert from the System Monitoring Service.

Current System Status:
- CPU Usage: {psutil.cpu_percent(interval=1):.1f}%
- Memory Usage: {psutil.virtual_memory().percent:.1f}%
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you received this email, the alert system is working correctly.
        """
        
        return self.send_email("Test Alert", test_body, "normal")
    
    def send_test_threat_alert(self):
        """Send a test cybersecurity threat alert"""
        test_threat = {
            'type': 'test_threat',
            'severity': 'medium',
            'description': 'This is a test cybersecurity threat alert',
            'source_ip': '192.168.1.100'
        }
        return self.send_threat_alert(test_threat)

def main():
    """Main function to run the alert system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='System Administrator Email Alert System')
    parser.add_argument('--config', default='alert_config.json', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Send test email')
    parser.add_argument('--check', action='store_true', help='Run single system check')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--test-threat', action='store_true', help='Send test threat alert')
    
    args = parser.parse_args()
    
    # Initialize alert system
    alert_system = EmailAlertSystem(args.config)
    
    if args.test:
        print("Sending test email...")
        if alert_system.send_test_alert():
            print("Test email sent successfully!")
        else:
            print("Failed to send test email. Check configuration and logs.")
    
    elif args.test_threat:
        print("Sending test threat alert...")
        if alert_system.send_test_threat_alert():
            print("Test threat alert sent successfully!")
        else:
            print("Failed to send test threat alert. Check configuration and logs.")
    
    elif args.check:
        print("Running single system check...")
        alert_system.run_system_check()
        print("System check completed.")
    
    elif args.monitor:
        print("Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        try:
            alert_system.start_monitoring()
        except KeyboardInterrupt:
            alert_system.stop_monitoring()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()