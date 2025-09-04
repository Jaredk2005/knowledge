export interface Incident {
  id: string;
  timestamp: Date;
  type: 'malware' | 'intrusion' | 'ddos' | 'phishing' | 'data_breach' | 'brute_force';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  target: string;
  description: string;
  status: 'detected' | 'investigating' | 'contained' | 'resolved';
  responseActions: string[];
  affectedSystems: string[];
}

export interface NetworkTraffic {
  timestamp: Date;
  source: string;
  destination: string;
  protocol: string;
  bytes: number;
  suspicious: boolean;
}

export interface SystemStatus {
  component: string;
  status: 'online' | 'offline' | 'warning' | 'error';
  lastCheck: Date;
  responseTime: number;
}

export interface Alert {
  id: string;
  timestamp: Date;
  message: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  acknowledged: boolean;
  correlationId?: string;
  sourceSystem: string;
  riskScore: number;
  isDuplicate: boolean;
  originalAlertId?: string;
  relatedAlerts: string[];
}

export interface ThreatDetection {
  id: string;
  timestamp: Date;
  threatType: 'behavioral_anomaly' | 'signature_match' | 'ml_detection' | 'correlation_rule';
  confidence: number;
  riskScore: number;
  indicators: string[];
  affectedAssets: string[];
  mitreTactics: string[];
  description: string;
}

export interface AnomalyDetection {
  id: string;
  timestamp: Date;
  anomalyType: 'traffic_spike' | 'unusual_login' | 'data_exfiltration' | 'privilege_escalation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  baseline: number;
  observed: number;
  deviation: number;
  source: string;
  description: string;
}