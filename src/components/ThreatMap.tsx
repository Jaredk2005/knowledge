import React, { useMemo } from 'react';
import { MapPin, AlertTriangle, Shield } from 'lucide-react';
import { Incident } from '../types/incident';
import { GlobalThreatMap } from './GlobalThreatMap';

interface ThreatMapProps {
  incidents: Incident[];
}

export function ThreatMap({ incidents }: ThreatMapProps) {
  const threatStats = useMemo(() => {
    const stats = {
      critical: incidents.filter(i => i.severity === 'critical').length,
      high: incidents.filter(i => i.severity === 'high').length,
      medium: incidents.filter(i => i.severity === 'medium').length,
      low: incidents.filter(i => i.severity === 'low').length,
    };
    return stats;
  }, [incidents]);

  const recentIncidents = incidents.slice(0, 8);

  return (
    <div className="space-y-6">
      {/* Enhanced Global Threat Map */}
      <GlobalThreatMap />
      
      {/* Legacy Local Incidents Map */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <MapPin className="h-6 w-6 text-blue-400 mr-2" />
            Local Incident Map
          </h2>
          <div className="flex space-x-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              <span className="text-gray-300">Critical ({threatStats.critical})</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-orange-500 rounded-full mr-2"></div>
              <span className="text-gray-300">High ({threatStats.high})</span>
            </div>
          </div>
        </div>

        {/* Simulated Local Network Map */}
        <div className="relative bg-gray-900 rounded-lg p-4 h-64 mb-6 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-gray-900 to-purple-900/20"></div>
          
          {/* Grid overlay */}
          <div className="absolute inset-0 opacity-20">
            <div className="grid grid-cols-12 grid-rows-8 h-full">
              {Array.from({ length: 96 }).map((_, i) => (
                <div key={i} className="border border-gray-600"></div>
              ))}
            </div>
          </div>

          {/* Local incident indicators */}
          {recentIncidents.map((incident, index) => (
            <div
              key={incident.id}
              className={`absolute w-4 h-4 rounded-full animate-pulse ${
              incident.severity === 'critical' ? 'bg-red-500' :
              incident.severity === 'high' ? 'bg-orange-500' :
              incident.severity === 'medium' ? 'bg-yellow-500' :
              'bg-blue-500'
              }`}
              style={{
                left: `${Math.random() * 90 + 5}%`,
                top: `${Math.random() * 80 + 10}%`,
                animationDelay: `${index * 0.5}s`
              }}
              title={`${incident.type} - ${incident.severity}`}
            >
              <div
                className={`absolute inset-0 rounded-full animate-ping ${
                  incident.severity === 'critical' ? 'bg-red-500' :
                  incident.severity === 'high' ? 'bg-orange-500' :
                  incident.severity === 'medium' ? 'bg-yellow-500' :
                  'bg-blue-500'
                }`}></div>
            </div>
          ))}
        </div>

        {/* Local Incident Summary */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-red-400 text-2xl font-bold">{threatStats.critical}</div>
            <div className="text-gray-400 text-sm">Critical Incidents</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-orange-400 text-2xl font-bold">{threatStats.high}</div>
            <div className="text-gray-400 text-sm">High Incidents</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-yellow-400 text-2xl font-bold">{threatStats.medium}</div>
            <div className="text-gray-400 text-sm">Medium Incidents</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-blue-400 text-2xl font-bold">{threatStats.low}</div>
            <div className="text-gray-400 text-sm">Low Incidents</div>
          </div>
        </div>
      </div>
    </div>
  );
}