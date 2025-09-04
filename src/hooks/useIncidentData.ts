import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabaseClient'
import { backendService, BackendThreat, BackendStats } from '../services/backendService'
import { criticalIncidentsService, CriticalIncident } from '../services/criticalIncidentsService'
import { Incident, NetworkTraffic, SystemStatus, Alert, ThreatDetection, AnomalyDetection } from '../types/incident'
import { generateSystemStatus, generateNetworkTraffic, generateAlert, generateThreatDetection, generateAnomalyDetection, correlateAlerts } from '../utils/dataSimulator'

import { fetchAlerts as fetchBackendAlerts } from '../services/backendApi'

// Function to transform database incident to app incident format
function transformDbIncidentToAppIncident(dbIncident: any): Incident {
  console.log('Transforming incident:', dbIncident)
  return {
    id: dbIncident.id,
    timestamp: new Date(dbIncident.created_at),
    type: dbIncident.incident_type as any,
    severity: dbIncident.severity as any,
    source: dbIncident.source_ip || dbIncident.source_system || 'Unknown',
    target: dbIncident.destination_ip || dbIncident.target_system || 'Unknown',
    description: dbIncident.description,
    status: dbIncident.status as any,
    responseActions: [], // Will be populated from incident_actions table if needed
    affectedSystems: dbIncident.affected_systems || []
  }
}

export function useIncidentData() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [networkTraffic, setNetworkTraffic] = useState<NetworkTraffic[]>([])
  const [systemStatus, setSystemStatus] = useState<SystemStatus[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [threatDetections, setThreatDetections] = useState<ThreatDetection[]>([])
  const [anomalies, setAnomalies] = useState<AnomalyDetection[]>([])
  const [isMonitoring, setIsMonitoring] = useState(true)
  const [backendConnected, setBackendConnected] = useState(false)
  const [backendStats, setBackendStats] = useState<BackendStats | null>(null)
  const [threatBackendConnected, setThreatBackendConnected] = useState(false)
  const [criticalIncidents, setCriticalIncidents] = useState<CriticalIncident[]>([])
  const [criticalIncidentsConnected, setCriticalIncidentsConnected] = useState(false)

  // Fetch incidents from database
  async function fetchIncidents() {
    try {
      const { data, error } = await supabase
        .from('incidents')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50)
      
      if (error) {
        console.error('Error fetching incidents from database:', error)
        return
      }
      
      console.log(`📊 Fetched ${data?.length || 0} incidents from database`)
      
      if (data) {
        const transformedIncidents = data.map(transformDbIncidentToAppIncident)
        console.log(`✅ Transformed ${transformedIncidents.length} incidents for display`)
        setIncidents(transformedIncidents)
      }
    } catch (error) {
      console.error('Failed to fetch incidents:', error)
    }
  }

  useEffect(() => {
    // Initialize with some default data
    setSystemStatus(generateSystemStatus())
    
    // Fetch real incidents from database
    fetchIncidents()
    
    // Set up periodic data generation for demo purposes
    const interval = setInterval(() => {
      if (isMonitoring) {
        // Generate some demo network traffic
        setNetworkTraffic(prev => {
          const newTraffic = generateNetworkTraffic()
          return [newTraffic, ...prev.slice(0, 99)] // Keep last 100 entries
        })
        
        // Occasionally generate alerts and threats
        if (Math.random() < 0.3) {
          const newAlert = generateAlert()
          setAlerts(prev => {
            const updated = [newAlert, ...prev.slice(0, 49)]
            return correlateAlerts(updated)
          })
        }
        
        if (Math.random() < 0.2) {
          const newThreat = generateThreatDetection()
          setThreatDetections(prev => [newThreat, ...prev.slice(0, 29)])
        }
        
        if (Math.random() < 0.15) {
          const newAnomaly = generateAnomalyDetection()
          setAnomalies(prev => [newAnomaly, ...prev.slice(0, 19)])
        }
      }
    }, 2000)

    // Set up backend service listeners
    const unsubscribeThreat = backendService.onThreatDetected((threat: BackendThreat) => {
      // Convert backend threat to incident
      const incident: Incident = {
        id: threat.id,
        timestamp: new Date(threat.timestamp),
        type: threat.threat_type as any,
        severity: threat.severity >= 8 ? 'critical' : threat.severity >= 6 ? 'high' : threat.severity >= 4 ? 'medium' : 'low',
        source: threat.source_ip,
        target: threat.destination_ip,
        description: threat.description,
        status: threat.blocked ? 'contained' : 'detected',
        responseActions: threat.blocked ? ['Block IP address', 'Notify security team'] : ['Investigate source'],
        affectedSystems: [threat.destination_ip]
      }
      
      setIncidents(prev => [incident, ...prev.slice(0, 49)])
      
      // Also create an alert
      const alert: Alert = {
        id: `ALT-${threat.id}`,
        timestamp: new Date(threat.timestamp),
        message: threat.description,
        type: threat.severity >= 8 ? 'critical' : threat.severity >= 6 ? 'error' : 'warning',
        acknowledged: false,
        sourceSystem: 'Backend Threat Detection',
        riskScore: Math.round(threat.confidence * 100),
        isDuplicate: false,
        relatedAlerts: []
      }
      
      setAlerts(prev => {
        const updated = [alert, ...prev.slice(0, 49)]
        return correlateAlerts(updated)
      })
      
      // Create threat detection entry
      const threatDetection: ThreatDetection = {
        id: `THR-${threat.id}`,
        timestamp: new Date(threat.timestamp),
        threatType: threat.threat_type.includes('behavioral') ? 'behavioral_anomaly' : 
                   threat.threat_type.includes('signature') ? 'signature_match' : 'ml_detection',
        confidence: Math.round(threat.confidence * 100),
        riskScore: Math.round(threat.confidence * 100),
        indicators: threat.indicators,
        affectedAssets: [threat.destination_ip],
        mitreTactics: ['Initial Access', 'Execution'],
        description: threat.description
      }
      
      setThreatDetections(prev => [threatDetection, ...prev.slice(0, 29)])
    })

    const unsubscribeStats = backendService.onStatsUpdate((stats: BackendStats) => {
      setBackendStats(stats)
    })

    const unsubscribeConnection = backendService.onConnectionChange((connected: boolean) => {
      setBackendConnected(connected)
    })

    const unsubscribeThreatBackend = backendService.onThreatBackendConnectionChange((connected: boolean) => {
      setThreatBackendConnected(connected)
    })

    // Set up critical incidents service listeners
    const unsubscribeCriticalIncident = criticalIncidentsService.onCriticalIncidentDetected((incident: CriticalIncident) => {
      setCriticalIncidents(prev => [incident, ...prev.slice(0, 49)])
      
      // Also add to regular incidents for dashboard display
      const regularIncident: Incident = {
        id: incident.id,
        timestamp: new Date(incident.detected_at),
        type: incident.incident_type as any,
        severity: incident.severity as any,
        source: incident.source_ip || incident.source_system || 'Unknown',
        target: incident.destination_ip || incident.target_system || 'Unknown',
        description: incident.description,
        status: incident.status as any,
        responseActions: ['Investigate immediately', 'Notify security team', 'Isolate affected systems'],
        affectedSystems: incident.affected_systems
      }
      
      setIncidents(prev => [regularIncident, ...prev.slice(0, 49)])
    })

    const unsubscribeCriticalConnection = criticalIncidentsService.onConnectionChange((connected: boolean) => {
      setCriticalIncidentsConnected(connected)
    })

    // Supabase setup (keep existing functionality)
    fetchAlerts()
    fetchAlertsFromBackend()

    // Set up real-time subscription for incidents
    const incidentsChannel = supabase.channel('public:incidents')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'incidents' }, (payload) => {
        console.log('Incident change detected:', payload)
        fetchIncidents() // Refetch incidents when changes occur
      })
      .subscribe()
    const channel = supabase.channel('public:alerts')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'alerts' }, (payload) => {
        fetchAlerts()
      })
      .subscribe()

    return () => {
      clearInterval(interval)
      unsubscribeThreat()
      unsubscribeStats()
      unsubscribeConnection()
      unsubscribeThreatBackend()
      unsubscribeCriticalIncident()
      unsubscribeCriticalConnection()
      
      try {
        supabase.removeChannel(incidentsChannel)
        supabase.removeChannel(channel)
      } catch (e) {
        // ignore
      }
    }
  }, [isMonitoring])

  async function fetchAlerts() {
    const { data, error } = await supabase.from('alerts').select('*').order('timestamp', { ascending: false })
    if (!error && data) setAlerts(data)
  }


  async function fetchAlertsFromBackend() {
    try {
      const items = await fetchBackendAlerts(50)
      if (items && items.length) {
        setAlerts(items)
      }
    } catch (e) {
      console.warn('Backend alerts fetch failed', e)
    }
  }
  const toggleMonitoring = () => {
    setIsMonitoring(!isMonitoring)
  }

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ))
  }

  const resolveIncident = (incidentId: string) => {
    // Update incident in database
    const updateIncidentInDb = async () => {
      try {
        const { error } = await supabase
          .from('incidents')
          .update({ status: 'resolved', resolved_at: new Date().toISOString() })
          .eq('id', incidentId)
        
        if (error) {
          console.error('Error updating incident:', error)
        } else {
          // Update local state
          setIncidents(prev => prev.map(incident => 
            incident.id === incidentId ? { ...incident, status: 'resolved' as const } : incident
          ))
        }
      } catch (error) {
        console.error('Failed to update incident:', error)
      }
    }
    
    updateIncidentInDb()
    setIncidents(prev => prev.map(incident => 
      incident.id === incidentId ? { ...incident, status: 'resolved' as const } : incident
    ))
  }

  return { 
    incidents,
    networkTraffic,
    systemStatus,
    alerts,
    threatDetections,
    anomalies,
    isMonitoring,
    toggleMonitoring,
    acknowledgeAlert,
    resolveIncident,
    backendConnected,
    threatBackendConnected,
    backendStats,
    criticalIncidents,
    criticalIncidentsConnected,
    blockIP: backendService.blockIP.bind(backendService),
    unblockIP: backendService.unblockIP.bind(backendService)
  }
}