import React from 'react';
import { Bell, CheckCircle, AlertTriangle, XCircle, Info } from 'lucide-react';
import { Alert } from '../types/incident';
import { useIncidentData } from '../hooks/useIncidentData';

interface AlertPanelProps {
  alerts: Alert[];
}

export function AlertPanel({ alerts }: AlertPanelProps) {
  const { acknowledgeAlert } = useIncidentData();

  // Filter out duplicate alerts for display
  const displayAlerts = alerts.filter(alert => !alert.isDuplicate);
  const correlatedAlerts = alerts.filter(alert => alert.correlationId && !alert.isDuplicate);

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'info':
        return <Info className="h-5 w-5 text-blue-400" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-400" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-400" />;
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Info className="h-5 w-5 text-blue-400" />;
    }
  };

  const getAlertBgColor = (type: string) => {
    switch (type) {
      case 'info':
        return 'bg-blue-900/30 border-blue-700/50';
      case 'warning':
        return 'bg-yellow-900/30 border-yellow-700/50';
      case 'error':
        return 'bg-red-900/30 border-red-700/50';
      case 'critical':
        return 'bg-red-900/50 border-red-600/50';
      default:
        return 'bg-blue-900/30 border-blue-700/50';
    }
  };

  const unacknowledgedCount = displayAlerts.filter(a => !a.acknowledged).length;
  const duplicateCount = alerts.filter(a => a.isDuplicate).length;

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Bell className="h-6 w-6 text-yellow-400 mr-2" />
          Active Alerts (Correlated)
        </h2>
        <div className="flex items-center space-x-2">
          {duplicateCount > 0 && (
            <div className="bg-blue-600 text-white text-xs px-2 py-1 rounded-full">
              {duplicateCount} deduplicated
            </div>
          )}
          {unacknowledgedCount > 0 && (
            <div className="bg-red-600 text-white text-xs px-2 py-1 rounded-full">
              {unacknowledgedCount} new
            </div>
          )}
        </div>
      </div>

      {/* Alert Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="text-red-400 text-lg font-bold">{displayAlerts.filter(a => a.type === 'critical').length}</div>
          <div className="text-gray-400 text-xs">Critical</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="text-orange-400 text-lg font-bold">{displayAlerts.filter(a => a.type === 'error').length}</div>
          <div className="text-gray-400 text-xs">Error</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="text-blue-400 text-lg font-bold">{correlatedAlerts.length}</div>
          <div className="text-gray-400 text-xs">Correlated</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="text-green-400 text-lg font-bold">{displayAlerts.filter(a => a.riskScore > 70).length}</div>
          <div className="text-gray-400 text-xs">High Risk</div>
          </div>
      </div>

      <div className="space-y-3 max-h-80 overflow-y-auto">
        {displayAlerts.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-3" />
            <p className="text-gray-400">No active alerts</p>
          </div>
        ) : (
          displayAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border ${getAlertBgColor(alert.type)} ${
                alert.acknowledged ? 'opacity-60' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <p className="text-white text-sm font-medium">{alert.message}</p>
                    <div className="flex items-center space-x-4 mt-1 text-xs text-gray-400">
                      <span>{alert.timestamp.toLocaleString()}</span>
                      <span>Source: {alert.sourceSystem}</span>
                      <span className={`font-medium ${
                        alert.riskScore > 70 ? 'text-red-400' : 
                        alert.riskScore > 40 ? 'text-yellow-400' : 'text-green-400'
                      }`}>
                        Risk: {alert.riskScore}
                      </span>
                    </div>
                    {alert.correlationId && (
                      <div className="mt-2">
                        <span className="text-xs px-2 py-1 bg-blue-900/50 text-blue-300 rounded">
                          Correlated ({alert.relatedAlerts.length + 1} alerts)
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                {!alert.acknowledged && (
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    className="text-gray-400 hover:text-white transition-colors text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
                  >
                    Ack
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}