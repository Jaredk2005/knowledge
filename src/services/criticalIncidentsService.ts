import { io, Socket } from 'socket.io-client';

export interface CriticalIncident {
  id: string;
  incident_number: string;
  title: string;
  description: string;
  incident_type: string;
  severity: string;
  status: string;
  source_ip?: string;
  destination_ip?: string;
  source_system?: string;
  target_system?: string;
  affected_systems: string[];
  indicators_of_compromise: string[];
  mitre_tactics: string[];
  confidence_score: number;
  risk_score: number;
  detected_at: string;
  first_seen?: string;
  last_seen?: string;
}

export interface CriticalIncidentStats {
  total_critical: number;
  critical_last_hour: number;
  active_critical: number;
  incident_types: Record<string, number>;
  avg_risk_score: number;
}

class CriticalIncidentsService {
  private socket: Socket | null = null;
  private backendUrl = 'http://localhost:8002';
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;

  // Event listeners
  private incidentListeners: Array<(incident: CriticalIncident) => void> = [];
  private resolvedListeners: Array<(incidentId: string) => void> = [];
  private connectionListeners: Array<(connected: boolean) => void> = [];

  constructor() {
    this.connect();
  }

  private connect() {
    try {
      this.socket = io(this.backendUrl, {
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true
      });

      this.socket.on('connect', () => {
        console.log('Connected to critical incidents backend');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.notifyConnectionListeners(true);
        
        // Subscribe to critical incident updates
        this.socket?.emit('subscribe_critical_incidents');
      });

      this.socket.on('disconnect', () => {
        console.log('Disconnected from critical incidents backend');
        this.isConnected = false;
        this.notifyConnectionListeners(false);
        this.attemptReconnect();
      });

      this.socket.on('critical_incident_detected', (incident: any) => {
        console.log('Critical incident detected:', incident);
        this.notifyIncidentListeners(incident);
      });

      this.socket.on('incident_resolved', (data: any) => {
        console.log('Incident resolved:', data);
        this.notifyResolvedListeners(data.incident_id);
      });

      this.socket.on('connect_error', (error) => {
        console.error('Critical incidents connection error:', error);
        this.isConnected = false;
        this.notifyConnectionListeners(false);
        this.attemptReconnect();
      });

    } catch (error) {
      console.error('Failed to initialize critical incidents connection:', error);
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect to critical incidents backend (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached. Critical incidents connection failed.');
    }
  }

  // Public methods for subscribing to events
  onCriticalIncidentDetected(callback: (incident: CriticalIncident) => void) {
    this.incidentListeners.push(callback);
    return () => {
      this.incidentListeners = this.incidentListeners.filter(cb => cb !== callback);
    };
  }

  onIncidentResolved(callback: (incidentId: string) => void) {
    this.resolvedListeners.push(callback);
    return () => {
      this.resolvedListeners = this.resolvedListeners.filter(cb => cb !== callback);
    };
  }

  onConnectionChange(callback: (connected: boolean) => void) {
    this.connectionListeners.push(callback);
    return () => {
      this.connectionListeners = this.connectionListeners.filter(cb => cb !== callback);
    };
  }

  // Notification methods
  private notifyIncidentListeners(incident: CriticalIncident) {
    this.incidentListeners.forEach(callback => {
      try {
        callback(incident);
      } catch (error) {
        console.error('Error in incident listener:', error);
      }
    });
  }

  private notifyResolvedListeners(incidentId: string) {
    this.resolvedListeners.forEach(callback => {
      try {
        callback(incidentId);
      } catch (error) {
        console.error('Error in resolved listener:', error);
      }
    });
  }

  private notifyConnectionListeners(connected: boolean) {
    this.connectionListeners.forEach(callback => {
      try {
        callback(connected);
      } catch (error) {
        console.error('Error in connection listener:', error);
      }
    });
  }

  // API methods
  async getCriticalIncidents(limit: number = 50): Promise<{ incidents: CriticalIncident[]; total: number }> {
    try {
      const response = await fetch(`${this.backendUrl}/api/critical-incidents?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching critical incidents:', error);
      return { incidents: [], total: 0 };
    }
  }

  async getCriticalIncidentStatistics(): Promise<CriticalIncidentStats> {
    try {
      const response = await fetch(`${this.backendUrl}/api/critical-incidents/statistics`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching critical incident statistics:', error);
      return {
        total_critical: 0,
        critical_last_hour: 0,
        active_critical: 0,
        incident_types: {},
        avg_risk_score: 0
      };
    }
  }

  async analyzeEvent(eventData: any): Promise<any> {
    try {
      const response = await fetch(`${this.backendUrl}/api/critical-incidents/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(eventData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error analyzing event:', error);
      throw error;
    }
  }

  async resolveIncident(incidentId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.backendUrl}/api/critical-incidents/${incidentId}/resolve`, {
        method: 'POST'
      });
      return response.ok;
    } catch (error) {
      console.error('Failed to resolve incident:', error);
      return false;
    }
  }

  isBackendConnected(): boolean {
    return this.isConnected;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
  }
}

// Export singleton instance
export const criticalIncidentsService = new CriticalIncidentsService();