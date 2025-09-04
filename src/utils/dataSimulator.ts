import { Incident, NetworkTraffic, SystemStatus, Alert, ThreatDetection, AnomalyDetection } from '../types/incident';

const incidentTypes = ['malware', 'intrusion', 'ddos', 'phishing', 'data_breach', 'brute_force'] as const;
const severityLevels = ['low', 'medium', 'high', 'critical'] as const;
const statusTypes = ['detected', 'investigating', 'contained', 'resolved'] as const;

const sampleSources = [
  '192.168.1.45',
  '10.0.0.23',
  '172.16.0.100',
  '203.0.113.5',
  '198.51.100.12',
  '192.0.2.146'
];

const sampleTargets = [
  'web-server-01',
  'database-primary',
  'mail-server',
  'file-server',
  'domain-controller',
  'api-gateway'
];

const responseActions = [
  'Block IP address',
  'Isolate affected system',
  'Update firewall rules',
  'Scan for malware',
  'Reset user credentials',
  'Enable DDoS protection',
  'Notify security team',
  'Create backup'
];

const sourceSystems = [
  'SIEM',
  'IDS/IPS',
  'Firewall',
  'Endpoint Protection',
  'Network Monitor',
  'Email Security',
  'Web Proxy',
  'DNS Monitor'
];

const mitreTactics = [
  'Initial Access',
  'Execution',
  'Persistence',
  'Privilege Escalation',
  'Defense Evasion',
  'Credential Access',
  'Discovery',
  'Lateral Movement',
  'Collection',
  'Exfiltration'
];

export function generateRandomIncident(): Incident {
  const type = incidentTypes[Math.floor(Math.random() * incidentTypes.length)];
  const severity = severityLevels[Math.floor(Math.random() * severityLevels.length)];
  const status = statusTypes[Math.floor(Math.random() * statusTypes.length)];
  
  return {
    id: `INC-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date(Date.now() - Math.random() * 86400000), // Last 24 hours
    type,
    severity,
    source: sampleSources[Math.floor(Math.random() * sampleSources.length)],
    target: sampleTargets[Math.floor(Math.random() * sampleTargets.length)],
    description: `${type.replace('_', ' ')} detected from ${sampleSources[Math.floor(Math.random() * sampleSources.length)]}`,
    status,
    responseActions: responseActions.slice(0, Math.floor(Math.random() * 3) + 1),
    affectedSystems: sampleTargets.slice(0, Math.floor(Math.random() * 2) + 1)
  };
}

export function generateNetworkTraffic(): NetworkTraffic {
  const suspicious = Math.random() < 0.15; // 15% chance of suspicious traffic
  
  return {
    timestamp: new Date(),
    source: sampleSources[Math.floor(Math.random() * sampleSources.length)],
    destination: sampleTargets[Math.floor(Math.random() * sampleTargets.length)],
    protocol: ['TCP', 'UDP', 'ICMP'][Math.floor(Math.random() * 3)],
    bytes: Math.floor(Math.random() * 10000) + 100,
    suspicious
  };
}

export function generateSystemStatus(): SystemStatus[] {
  const components = [
    'Firewall',
    'IDS/IPS',
    'Antivirus',
    'Web Application Firewall',
    'SIEM',
    'Endpoint Protection',
    'Network Monitor',
    'Threat Intelligence'
  ];

  return components.map(component => ({
    component,
    status: Math.random() < 0.9 ? 'online' : ['offline', 'warning', 'error'][Math.floor(Math.random() * 3)] as any,
    lastCheck: new Date(Date.now() - Math.random() * 300000), // Last 5 minutes
    responseTime: Math.floor(Math.random() * 200) + 10
  }));
}

export function generateAlert(): Alert {
  const messages = [
    'Suspicious login attempt detected',
    'Malware signature found in email attachment',
    'Unusual network traffic pattern observed',
    'Failed authentication attempts exceed threshold',
    'Potential data exfiltration detected',
    'System vulnerability scan completed',
    'Anomalous user behavior detected',
    'Multiple failed login attempts from same IP',
    'Suspicious file execution detected',
    'Unauthorized privilege escalation attempt'
  ];

  const types = ['info', 'warning', 'error', 'critical'] as const;
  
  return {
    id: `ALT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date(),
    message: messages[Math.floor(Math.random() * messages.length)],
    type: types[Math.floor(Math.random() * types.length)],
    acknowledged: false,
    sourceSystem: sourceSystems[Math.floor(Math.random() * sourceSystems.length)],
    riskScore: Math.floor(Math.random() * 100) + 1,
    isDuplicate: false,
    relatedAlerts: []
  };
}

export function generateThreatDetection(): ThreatDetection {
  const threatTypes = ['behavioral_anomaly', 'signature_match', 'ml_detection', 'correlation_rule'] as const;
  const indicators = [
    'Suspicious process execution',
    'Unusual network connections',
    'File system modifications',
    'Registry changes',
    'Memory injection detected',
    'Command line anomalies',
    'DNS tunneling activity',
    'Encrypted traffic to suspicious domains'
  ];

  return {
    id: `THR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date(),
    threatType: threatTypes[Math.floor(Math.random() * threatTypes.length)],
    confidence: Math.floor(Math.random() * 40) + 60, // 60-100%
    riskScore: Math.floor(Math.random() * 100) + 1,
    indicators: indicators.slice(0, Math.floor(Math.random() * 3) + 1),
    affectedAssets: sampleTargets.slice(0, Math.floor(Math.random() * 2) + 1),
    mitreTactics: mitreTactics.slice(0, Math.floor(Math.random() * 2) + 1),
    description: `Advanced threat detected using ${threatTypes[Math.floor(Math.random() * threatTypes.length)].replace('_', ' ')}`
  };
}

export function generateAnomalyDetection(): AnomalyDetection {
  const anomalyTypes = ['traffic_spike', 'unusual_login', 'data_exfiltration', 'privilege_escalation'] as const;
  const type = anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)];
  const baseline = Math.floor(Math.random() * 1000) + 100;
  const observed = baseline + Math.floor(Math.random() * 2000) + 500;
  
  return {
    id: `ANO-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date(),
    anomalyType: type,
    severity: severityLevels[Math.floor(Math.random() * severityLevels.length)],
    baseline,
    observed,
    deviation: Math.round(((observed - baseline) / baseline) * 100),
    source: sampleSources[Math.floor(Math.random() * sampleSources.length)],
    description: `${type.replace('_', ' ')} anomaly detected - ${Math.round(((observed - baseline) / baseline) * 100)}% above baseline`
  };
}

// Alert correlation and deduplication logic
export function correlateAlerts(alerts: Alert[]): Alert[] {
  const correlatedAlerts: Alert[] = [];
  const processedIds = new Set<string>();
  
  alerts.forEach(alert => {
    if (processedIds.has(alert.id)) return;
    
    // Find similar alerts within 5 minutes
    const similarAlerts = alerts.filter(other => 
      other.id !== alert.id &&
      !processedIds.has(other.id) &&
      Math.abs(other.timestamp.getTime() - alert.timestamp.getTime()) < 300000 && // 5 minutes
      (
        other.message.includes(alert.message.split(' ')[0]) ||
        other.sourceSystem === alert.sourceSystem ||
        (alert.message.includes('login') && other.message.includes('login')) ||
        (alert.message.includes('malware') && other.message.includes('malware'))
      )
    );
    
    if (similarAlerts.length > 0) {
      // Create correlation
      const correlationId = `CORR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const relatedIds = similarAlerts.map(a => a.id);
      
      // Mark primary alert
      const primaryAlert = {
        ...alert,
        correlationId,
        relatedAlerts: relatedIds,
        riskScore: Math.max(alert.riskScore, ...similarAlerts.map(a => a.riskScore))
      };
      
      correlatedAlerts.push(primaryAlert);
      processedIds.add(alert.id);
      
      // Mark related alerts as duplicates
      similarAlerts.forEach(similar => {
        correlatedAlerts.push({
          ...similar,
          isDuplicate: true,
          originalAlertId: alert.id,
          correlationId
        });
        processedIds.add(similar.id);
      });
    } else {
      correlatedAlerts.push(alert);
      processedIds.add(alert.id);
    }
  });
  
  return correlatedAlerts;
}