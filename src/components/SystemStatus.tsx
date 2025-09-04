import React from 'react';
import { Server, CheckCircle, AlertCircle, XCircle, Clock } from 'lucide-react';
import { SystemStatus as SystemStatusType } from '../types/incident';

interface SystemStatusProps {
  systems: SystemStatusType[];
}

export function SystemStatus({ systems }: SystemStatusProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'warning':
        return <AlertCircle className="h-5 w-5 text-yellow-400" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-400" />;
      case 'offline':
        return <XCircle className="h-5 w-5 text-gray-400" />;
      default:
        return <CheckCircle className="h-5 w-5 text-green-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'text-green-400';
      case 'warning':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      case 'offline':
        return 'text-gray-400';
      default:
        return 'text-green-400';
    }
  };

  const onlineCount = systems.filter(s => s.status === 'online').length;

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Server className="h-6 w-6 text-purple-400 mr-2" />
          System Status
        </h2>
        <div className="text-sm text-gray-400">
          {onlineCount}/{systems.length} Online
        </div>
      </div>

      <div className="space-y-3">
        {systems.map((system, index) => (
          <div key={index} className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
            <div className="flex items-center space-x-3">
              {getStatusIcon(system.status)}
              <div>
                <div className="text-white text-sm font-medium">{system.component}</div>
                <div className="flex items-center space-x-2 text-xs text-gray-400">
                  <Clock className="h-3 w-3" />
                  <span>Last check: {system.lastCheck.toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className={`text-sm font-medium ${getStatusColor(system.status)}`}>
                {system.status.charAt(0).toUpperCase() + system.status.slice(1)}
              </div>
              <div className="text-xs text-gray-400">
                {system.responseTime}ms
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}