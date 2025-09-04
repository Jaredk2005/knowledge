import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, Clock, CheckCircle, XCircle, Eye, Target, Activity } from 'lucide-react';
import { criticalIncidentsService, CriticalIncident, CriticalIncidentStats } from '../services/criticalIncidentsService';

export function CriticalIncidentsPanel() {
  const [incidents, setIncidents] = useState<CriticalIncident[]>([]);
  const [stats, setStats] = useState<CriticalIncidentStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedIncident, setSelectedIncident] = useState<CriticalIncident | null>(null);

  useEffect(() => {
    // Load initial data
    loadCriticalIncidents();
    loadStatistics();

    // Set up event listeners
    const unsubscribeIncident = criticalIncidentsService.onCriticalIncidentDetected((incident) => {
      setIncidents(prev => [incident, ...prev.slice(0, 49)]);
      loadStatistics(); // Refresh stats
    });

    const unsubscribeResolved = criticalIncidentsService.onIncidentResolved((incidentId) => {
      setIncidents(prev => prev.map(incident => 
        incident.id === incidentId 
          ? { ...incident, status: 'resolved' }
          : incident
      ));
      loadStatistics(); // Refresh stats
    });

    const unsubscribeConnection = criticalIncidentsService.onConnectionChange((connected) => {
      setIsConnected(connected);
    });

    // Set up periodic refresh
    const refreshInterval = setInterval(() => {
      loadCriticalIncidents();
      loadStatistics();
    }, 30000); // Refresh every 30 seconds

    return () => {
      unsubscribeIncident();
      unsubscribeResolved();
      unsubscribeConnection();
      clearInterval(refreshInterval);
    };
  }, []);

  const loadCriticalIncidents = async () => {
    try {
      setIsLoading(true);
      const response = await criticalIncidentsService.getCriticalIncidents(50);
      setIncidents(response.incidents);
    } catch (error) {
      console.error('Failed to load critical incidents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const statistics = await criticalIncidentsService.getCriticalIncidentStatistics();
      setStats(statistics);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    }
  };

  const handleResolveIncident = async (incidentId: string) => {
    try {
      const success = await criticalIncidentsService.resolveIncident(incidentId);
      if (success) {
        setIncidents(prev => prev.map(incident => 
          incident.id === incidentId 
            ? { ...incident, status: 'resolved' }
            : incident
        ));
        loadStatistics();
      }
    } catch (error) {
      console.error('Failed to resolve incident:', error);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-400 bg-red-900/30 border-red-700/50';
      case 'high':
        return 'text-orange-400 bg-orange-900/30 border-orange-700/50';
      case 'medium':
        return 'text-yellow-400 bg-yellow-900/30 border-yellow-700/50';
      case 'low':
        return 'text-blue-400 bg-blue-900/30 border-blue-700/50';
      default:
        return 'text-gray-400 bg-gray-900/30 border-gray-700/50';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'detected':
        return 'text-red-400';
      case 'investigating':
        return 'text-yellow-400';
      case 'contained':
        return 'text-blue-400';
      case 'resolved':
        return 'text-green-400';
      default:
        return 'text-gray-400';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <AlertTriangle className="h-6 w-6 text-red-400 mr-2" />
            Critical Incidents
          </h2>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-300">
                {isConnected ? 'Live Detection' : 'Offline'}
              </span>
            </div>
            <button
              onClick={() => { loadCriticalIncidents(); loadStatistics(); }}
              disabled={isLoading}
              className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 rounded-lg transition-colors"
            >
              <Activity className={`h-4 w-4 text-white ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Statistics */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-red-900/30 rounded-lg p-4 border border-red-700/50">
              <div className="text-red-400 text-2xl font-bold">{stats.total_critical}</div>
              <div className="text-gray-400 text-sm">Total Critical</div>
            </div>
            <div className="bg-orange-900/30 rounded-lg p-4 border border-orange-700/50">
              <div className="text-orange-400 text-2xl font-bold">{stats.critical_last_hour}</div>
              <div className="text-gray-400 text-sm">Last Hour</div>
            </div>
            <div className="bg-yellow-900/30 rounded-lg p-4 border border-yellow-700/50">
              <div className="text-yellow-400 text-2xl font-bold">{stats.active_critical}</div>
              <div className="text-gray-400 text-sm">Active</div>
            </div>
            <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-700/50">
              <div className="text-purple-400 text-2xl font-bold">{Math.round(stats.avg_risk_score)}</div>
              <div className="text-gray-400 text-sm">Avg Risk Score</div>
            </div>
          </div>
        )}

        {/* Incident Types Distribution */}
        {stats && Object.keys(stats.incident_types).length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Incident Types (24h)</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(stats.incident_types).map(([type, count]) => (
                <div key={type} className="bg-gray-900 rounded-lg p-3">
                  <div className="text-white font-medium">{count}</div>
                  <div className="text-gray-400 text-sm capitalize">{type.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Incidents List */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Critical Incidents</h3>
        
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-400"></div>
            <span className="ml-2 text-gray-400">Loading incidents...</span>
          </div>
        ) : incidents.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-3" />
            <p className="text-gray-400">No critical incidents detected</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {incidents.map((incident) => (
              <div
                key={incident.id}
                className={`p-4 rounded-lg border ${getSeverityColor(incident.severity)} cursor-pointer hover:bg-opacity-80 transition-colors`}
                onClick={() => setSelectedIncident(incident)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-xs font-mono bg-gray-700 px-2 py-1 rounded">
                        {incident.incident_number}
                      </span>
                      <span className={`text-xs font-medium ${getStatusColor(incident.status)}`}>
                        {incident.status.toUpperCase()}
                      </span>
                    </div>
                    <h4 className="text-white font-medium mb-1">{incident.title}</h4>
                    <p className="text-gray-300 text-sm mb-2">{incident.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      <span className="flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {formatTimestamp(incident.detected_at)}
                      </span>
                      {incident.source_ip && (
                        <span className="flex items-center">
                          <Target className="h-3 w-3 mr-1" />
                          {incident.source_ip}
                        </span>
                      )}
                      <span className="flex items-center">
                        <Shield className="h-3 w-3 mr-1" />
                        Risk: {incident.risk_score}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedIncident(incident);
                      }}
                      className="p-1 text-gray-400 hover:text-white transition-colors"
                      title="View details"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    {incident.status !== 'resolved' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleResolveIncident(incident.id);
                        }}
                        className="text-xs px-2 py-1 bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
                      >
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
                
                {/* MITRE Tactics */}
                {incident.mitre_tactics.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <p className="text-xs text-gray-400 mb-2">MITRE ATT&CK Tactics:</p>
                    <div className="flex flex-wrap gap-1">
                      {incident.mitre_tactics.map((tactic, index) => (
                        <span
                          key={index}
                          className="text-xs px-2 py-1 bg-red-900/50 text-red-300 rounded"
                        >
                          {tactic}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Incident Details Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-gray-700">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white">Incident Details</h3>
              <button
                onClick={() => setSelectedIncident(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <XCircle className="h-6 w-6" />
              </button>
            </div>
            
            <div className="space-y-6">
              {/* Basic Information */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Basic Information</h4>
                  <div className="space-y-2 text-sm">
                    <div><span className="text-gray-400">Incident Number:</span> <span className="text-white font-mono">{selectedIncident.incident_number}</span></div>
                    <div><span className="text-gray-400">Type:</span> <span className="text-white capitalize">{selectedIncident.incident_type.replace('_', ' ')}</span></div>
                    <div><span className="text-gray-400">Severity:</span> <span className={`font-medium ${getSeverityColor(selectedIncident.severity).split(' ')[0]}`}>{selectedIncident.severity.toUpperCase()}</span></div>
                    <div><span className="text-gray-400">Status:</span> <span className={`font-medium ${getStatusColor(selectedIncident.status)}`}>{selectedIncident.status.toUpperCase()}</span></div>
                    <div><span className="text-gray-400">Risk Score:</span> <span className="text-white">{selectedIncident.risk_score}/100</span></div>
                    <div><span className="text-gray-400">Confidence:</span> <span className="text-white">{selectedIncident.confidence_score}%</span></div>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Network Information</h4>
                  <div className="space-y-2 text-sm">
                    {selectedIncident.source_ip && <div><span className="text-gray-400">Source IP:</span> <span className="text-white font-mono">{selectedIncident.source_ip}</span></div>}
                    {selectedIncident.destination_ip && <div><span className="text-gray-400">Destination IP:</span> <span className="text-white font-mono">{selectedIncident.destination_ip}</span></div>}
                    {selectedIncident.source_system && <div><span className="text-gray-400">Source System:</span> <span className="text-white">{selectedIncident.source_system}</span></div>}
                    {selectedIncident.target_system && <div><span className="text-gray-400">Target System:</span> <span className="text-white">{selectedIncident.target_system}</span></div>}
                    <div><span className="text-gray-400">Detected:</span> <span className="text-white">{formatTimestamp(selectedIncident.detected_at)}</span></div>
                    {selectedIncident.first_seen && <div><span className="text-gray-400">First Seen:</span> <span className="text-white">{formatTimestamp(selectedIncident.first_seen)}</span></div>}
                  </div>
                </div>
              </div>
              
              {/* Description */}
              <div>
                <h4 className="text-lg font-semibold text-white mb-3">Description</h4>
                <p className="text-gray-300 bg-gray-900 p-4 rounded-lg">{selectedIncident.description}</p>
              </div>
              
              {/* Affected Systems */}
              {selectedIncident.affected_systems.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Affected Systems</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedIncident.affected_systems.map((system, index) => (
                      <span key={index} className="px-3 py-1 bg-red-900/30 text-red-300 rounded-lg text-sm">
                        {system}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Indicators of Compromise */}
              {selectedIncident.indicators_of_compromise.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">Indicators of Compromise</h4>
                  <div className="space-y-2">
                    {selectedIncident.indicators_of_compromise.map((ioc, index) => (
                      <div key={index} className="bg-gray-900 p-3 rounded-lg">
                        <span className="text-gray-300 font-mono text-sm">{ioc}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* MITRE ATT&CK Tactics */}
              {selectedIncident.mitre_tactics.length > 0 && (
                <div>
                  <h4 className="text-lg font-semibold text-white mb-3">MITRE ATT&CK Tactics</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedIncident.mitre_tactics.map((tactic, index) => (
                      <span key={index} className="px-3 py-1 bg-blue-900/30 text-blue-300 rounded-lg text-sm">
                        {tactic}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Actions */}
              <div className="flex space-x-3 pt-4 border-t border-gray-700">
                {selectedIncident.status !== 'resolved' && (
                  <button
                    onClick={() => {
                      handleResolveIncident(selectedIncident.id);
                      setSelectedIncident(null);
                    }}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                  >
                    Resolve Incident
                  </button>
                )}
                <button
                  onClick={() => setSelectedIncident(null)}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}