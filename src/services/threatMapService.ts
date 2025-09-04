import { supabase } from '../lib/supabaseClient';

export interface GlobalThreat {
  id: string;
  source_ip: string;
  destination_ip?: string;
  threat_type: string;
  severity: number;
  confidence: number;
  description: string;
  attack_vector: string;
  first_seen: string;
  last_seen: string;
  event_count: number;
  source_location: {
    country: string;
    country_code: string;
    city: string;
    latitude: number;
    longitude: number;
  };
  destination_location?: {
    country: string;
    country_code: string;
    city: string;
    latitude: number;
    longitude: number;
  };
}

export interface ThreatStatistics {
  overview: {
    total_threats: number;
    threats_last_hour: number;
    threats_last_24h: number;
    active_threats: number;
    avg_severity: number;
    countries_affected: number;
  };
  threat_types: Array<{
    type: string;
    count: number;
  }>;
  top_countries: Array<{
    country: string;
    country_code: string;
    count: number;
  }>;
  severity_distribution: Array<{
    severity: number;
    count: number;
  }>;
}

export interface ThreatMapFilters {
  hours?: number;
  min_severity?: number;
  country?: string;
  limit?: number;
}

class ThreatMapService {
  private baseUrl = 'http://localhost:8001/api';
  private wsUrl = 'ws://localhost:8001/ws/threats';
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;

  // Event listeners
  private threatListeners: Array<(threat: any) => void> = [];
  private connectionListeners: Array<(connected: boolean) => void> = [];

  constructor() {
    this.connectWebSocket();
  }

  // WebSocket connection management
  private connectWebSocket() {
    try {
      this.websocket = new WebSocket(this.wsUrl);

      this.websocket.onopen = () => {
        console.log('Connected to threat map WebSocket');
        this.reconnectAttempts = 0;
        this.notifyConnectionListeners(true);
      };

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'threat_detected') {
            this.notifyThreatListeners(data.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.websocket.onclose = () => {
        console.log('Disconnected from threat map WebSocket');
        this.notifyConnectionListeners(false);
        this.attemptReconnect();
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.notifyConnectionListeners(false);
      };

    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect to WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connectWebSocket();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max WebSocket reconnection attempts reached');
    }
  }

  // Event listener management
  onThreatDetected(callback: (threat: any) => void) {
    this.threatListeners.push(callback);
    return () => {
      this.threatListeners = this.threatListeners.filter(cb => cb !== callback);
    };
  }

  onConnectionChange(callback: (connected: boolean) => void) {
    this.connectionListeners.push(callback);
    return () => {
      this.connectionListeners = this.connectionListeners.filter(cb => cb !== callback);
    };
  }

  private notifyThreatListeners(threat: any) {
    this.threatListeners.forEach(callback => {
      try {
        callback(threat);
      } catch (error) {
        console.error('Error in threat listener:', error);
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
  async getGlobalThreats(filters: ThreatMapFilters = {}): Promise<{ threats: GlobalThreat[]; total: number }> {
    try {
      const params = new URLSearchParams();
      
      if (filters.hours) params.append('hours', filters.hours.toString());
      if (filters.min_severity) params.append('min_severity', filters.min_severity.toString());
      if (filters.country) params.append('country', filters.country);
      if (filters.limit) params.append('limit', filters.limit.toString());

      const response = await fetch(`${this.baseUrl}/threats/global?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching global threats:', error);
      
      // Fallback to Supabase if backend is unavailable
      return this.getThreatsFromSupabase(filters);
    }
  }

  async getThreatStatistics(): Promise<ThreatStatistics> {
    try {
      const response = await fetch(`${this.baseUrl}/threats/statistics`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching threat statistics:', error);
      
      // Return fallback statistics
      return {
        overview: {
          total_threats: 0,
          threats_last_hour: 0,
          threats_last_24h: 0,
          active_threats: 0,
          avg_severity: 0,
          countries_affected: 0
        },
        threat_types: [],
        top_countries: [],
        severity_distribution: []
      };
    }
  }

  async ingestThreat(threat: Partial<GlobalThreat>): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/threats/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(threat),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error ingesting threat:', error);
      throw error;
    }
  }

  async bulkIngestThreats(threats: Partial<GlobalThreat>[]): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/threats/bulk-ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(threats),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error bulk ingesting threats:', error);
      throw error;
    }
  }

  // Fallback to Supabase when backend is unavailable
  private async getThreatsFromSupabase(filters: ThreatMapFilters): Promise<{ threats: GlobalThreat[]; total: number }> {
    try {
      let query = supabase
        .from('threat_intelligence')
        .select('*')
        .eq('is_active', true)
        .order('first_seen', { ascending: false });

      if (filters.hours) {
        const hoursAgo = new Date(Date.now() - filters.hours * 60 * 60 * 1000).toISOString();
        query = query.gte('first_seen', hoursAgo);
      }

      if (filters.min_severity) {
        query = query.eq('severity', filters.min_severity === 1 ? 'low' : 
                                    filters.min_severity === 2 ? 'medium' : 
                                    filters.min_severity === 3 ? 'high' : 'critical');
      }

      if (filters.country) {
        query = query.ilike('description', `%${filters.country}%`);
      }

      if (filters.limit) {
        query = query.limit(filters.limit);
      }

      const { data, error } = await query;

      if (error) {
        throw error;
      }

      const threats: GlobalThreat[] = (data || []).map(row => ({
        id: row.id,
        source_ip: row.indicator_value,
        destination_ip: '',
        threat_type: row.threat_type,
        severity: row.severity === 'low' ? 1 : row.severity === 'medium' ? 5 : row.severity === 'high' ? 8 : 10,
        confidence: row.confidence,
        description: row.description,
        attack_vector: row.threat_type,
        first_seen: row.first_seen,
        last_seen: row.last_seen,
        event_count: 1,
        source_location: {
          country: 'Unknown',
          country_code: 'XX',
          city: 'Unknown',
          latitude: 0,
          longitude: 0
        },
        destination_location: undefined
      }));

      return { threats, total: threats.length };
    } catch (error) {
      console.error('Error fetching threats from Supabase:', error);
      return { threats: [], total: 0 };
    }
  }

  // Utility methods
  getSeverityColor(severity: number): string {
    const colors = {
      1: '#4CAF50', 2: '#8BC34A', 3: '#CDDC39', 4: '#FFEB3B', 5: '#FFC107',
      6: '#FF9800', 7: '#FF5722', 8: '#F44336', 9: '#E91E63', 10: '#9C27B0'
    };
    return colors[severity as keyof typeof colors] || '#9E9E9E';
  }

  getThreatTypeIcon(threatType: string): string {
    const icons = {
      'malware': '🦠',
      'botnet': '🤖',
      'ddos': '💥',
      'brute_force': '🔨',
      'sql_injection': '💉',
      'xss': '🕷️',
      'phishing': '🎣',
      'ransomware': '🔒',
      'apt': '🎯'
    };
    return icons[threatType as keyof typeof icons] || '⚠️';
  }

  formatThreatDescription(threat: GlobalThreat): string {
    return `${this.getThreatTypeIcon(threat.threat_type)} ${threat.description} (Severity: ${threat.severity}/10)`;
  }

  // Cleanup
  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }
}

export const threatMapService = new ThreatMapService();