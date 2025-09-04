import React from 'react';
import { Shield, Clock, CheckCircle } from 'lucide-react';
import { Incident } from '../types/incident';
import { useIncidentData } from '../hooks/useIncidentData';

interface IncidentListProps {
  incidents: Incident[];
}

export function IncidentList({ incidents }: IncidentListProps) {
  const { resolveIncident } = useIncidentData();

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-400 bg-red-900/30';
      case 'high':
        return 'text-orange-400 bg-orange-900/30';
      case 'medium':
        return 'text-yellow-400 bg-yellow-900/30';
      case 'low':
        return 'text-blue-400 bg-blue-900/30';
      default:
        return 'text-gray-400 bg-gray-900/30';
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

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Shield className="h-6 w-6 text-red-400 mr-2" />
          Recent Incidents
        </h2>
      </div>

      <div className="space-y-3">
        {incidents.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-3" />
            <p className="text-gray-400">No recent incidents</p>
          </div>
        ) : (
          incidents.map((incident) => (
            <div key={incident.id} className="bg-gray-900 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(incident.severity)}`}>
                      {incident.severity.toUpperCase()}
                    </span>
                    <span className="text-gray-400 text-xs">
                      {incident.type.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <p className="text-white text-sm font-medium">{incident.description}</p>
                  <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                    <span>Source: {incident.source}</span>
                    <span>Target: {incident.target}</span>
                    <span className="flex items-center">
                      <Clock className="h-3 w-3 mr-1" />
                      {incident.timestamp.toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`text-xs font-medium ${getStatusColor(incident.status)}`}>
                    {incident.status.charAt(0).toUpperCase() + incident.status.slice(1)}
                  </span>
                  {incident.status !== 'resolved' && (
                    <button
                      onClick={() => resolveIncident(incident.id)}
                      className="text-xs px-2 py-1 bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
                    >
                      Resolve
                    </button>
                  )}
                </div>
              </div>
              
              {incident.responseActions.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-400 mb-2">Response Actions:</p>
                  <div className="flex flex-wrap gap-1">
                    {incident.responseActions.map((action, index) => (
                      <span
                        key={index}
                        className="text-xs px-2 py-1 bg-blue-900/50 text-blue-300 rounded"
                      >
                        {action}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}