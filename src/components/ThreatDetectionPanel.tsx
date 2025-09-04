import React from 'react';
import { Shield, Brain, Target, AlertTriangle, TrendingUp } from 'lucide-react';
import { ThreatDetection, AnomalyDetection } from '../types/incident';
import { useIncidentData } from '../hooks/useIncidentData';

interface ThreatDetectionPanelProps {
  threatDetections: ThreatDetection[];
  anomalies: AnomalyDetection[];
}

export function ThreatDetectionPanel({ threatDetections, anomalies }: ThreatDetectionPanelProps) {
  const { backendStats, backendConnected, blockIP } = useIncidentData();
  const highConfidenceThreats = threatDetections.filter(t => t.confidence > 80);
  const criticalAnomalies = anomalies.filter(a => a.severity === 'critical' || a.deviation > 300);

  const getThreatTypeIcon = (type: string) => {
    switch (type) {
      case 'behavioral_anomaly':
        return <Brain className="h-5 w-5 text-purple-400" />;
      case 'signature_match':
        return <Target className="h-5 w-5 text-red-400" />;
      case 'ml_detection':
        return <Brain className="h-5 w-5 text-blue-400" />;
      case 'correlation_rule':
        return <AlertTriangle className="h-5 w-5 text-orange-400" />;
      default:
        return <Shield className="h-5 w-5 text-gray-400" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-red-400';
    if (confidence >= 70) return 'text-orange-400';
    if (confidence >= 50) return 'text-yellow-400';
    return 'text-blue-400';
  };

  const getAnomalyColor = (severity: string) => {
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

  return (
    <div className="space-y-6">
      {/* Threat Detection Overview */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Shield className="h-6 w-6 text-red-400 mr-2" />
            Advanced Threat Detection
          </h2>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
            <span className="text-red-400 text-sm">Active Scanning</span>
            {backendConnected && (
              <>
                <div className="w-2 h-2 bg-blue-400 rounded-full ml-2"></div>
                <span className="text-blue-400 text-sm">Backend Connected</span>
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-red-400 text-2xl font-bold">
              {backendStats?.active_threats || highConfidenceThreats.length}
            </div>
            <div className="text-gray-400 text-sm">High Confidence</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-orange-400 text-2xl font-bold">
              {backendStats?.total_threats || threatDetections.filter(t => t.threatType === 'ml_detection').length}
            </div>
            <div className="text-gray-400 text-sm">Total Threats</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-purple-400 text-2xl font-bold">
              {backendStats?.blocked_ips || threatDetections.filter(t => t.threatType === 'behavioral_anomaly').length}
            </div>
            <div className="text-gray-400 text-sm">Blocked IPs</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-blue-400 text-2xl font-bold">
              {backendStats?.threats_last_hour || criticalAnomalies.length}
            </div>
            <div className="text-gray-400 text-sm">Last Hour</div>
          </div>
        </div>

        {/* Backend Statistics */}
        {backendConnected && backendStats && (
          <div className="mb-6 p-4 bg-blue-900/20 border border-blue-700/50 rounded-lg">
            <h4 className="text-sm font-medium text-blue-300 mb-2">Live Backend Statistics</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-gray-400">24h Threats:</span>
                <span className="text-white ml-1">{backendStats.threats_last_24h}</span>
              </div>
              <div>
                <span className="text-gray-400">Correlations:</span>
                <span className="text-white ml-1">{backendStats.correlations?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-400">Top Threat IPs:</span>
                <span className="text-white ml-1">{backendStats.top_threat_ips?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-400">Threat Types:</span>
                <span className="text-white ml-1">{Object.keys(backendStats.threat_types || {}).length}</span>
              </div>
            </div>
          </div>
        )}

        {/* Recent Threat Detections */}
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Recent Threat Detections</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {threatDetections.slice(0, 8).map((threat) => (
              <div key={threat.id} className="bg-gray-900 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    {getThreatTypeIcon(threat.threatType)}
                    <div>
                      <div className="text-white text-sm font-medium">{threat.description}</div>
                      <div className="text-gray-400 text-xs mt-1">
                        {threat.timestamp.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${getConfidenceColor(threat.confidence)}`}>
                      {threat.confidence}% confidence
                    </div>
                    <div className="text-xs text-gray-400">Risk: {threat.riskScore}</div>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1 mt-2">
                  {threat.mitreTactics.map((tactic, index) => (
                    <span
                      key={index}
                      className="text-xs px-2 py-1 bg-red-900/50 text-red-300 rounded"
                    >
                      {tactic}
                    </span>
                  ))}
                </div>
                
                {threat.indicators.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-700">
                    <p className="text-xs text-gray-400 mb-1">Indicators:</p>
                    <div className="flex flex-wrap gap-1">
                      {threat.indicators.map((indicator, index) => (
                        <span
                          key={index}
                          className="text-xs px-2 py-1 bg-blue-900/50 text-blue-300 rounded"
                        >
                          {indicator}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Action buttons for backend threats */}
                {backendConnected && (
                  <div className="mt-2 pt-2 border-t border-gray-700">
                    <button
                      onClick={() => blockIP(threat.affectedAssets[0] || 'unknown')}
                      className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded mr-2"
                    >
                      Block IP
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Anomaly Detection */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <TrendingUp className="h-6 w-6 text-purple-400 mr-2" />
            Anomaly Detection
          </h2>
        </div>

        <div className="space-y-3">
          {anomalies.slice(0, 6).map((anomaly) => (
            <div key={anomaly.id} className="bg-gray-900 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getAnomalyColor(anomaly.severity)}`}>
                      {anomaly.severity.toUpperCase()}
                    </span>
                    <span className="text-gray-400 text-xs">
                      {anomaly.anomalyType.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <p className="text-white text-sm font-medium">{anomaly.description}</p>
                  <div className="flex items-center space-x-4 mt-2 text-xs text-gray-400">
                    <span>Source: {anomaly.source}</span>
                    <span>Baseline: {anomaly.baseline}</span>
                    <span>Observed: {anomaly.observed}</span>
                    <span className="text-red-400">+{anomaly.deviation}%</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-red-400">
                    {anomaly.deviation}% deviation
                  </div>
                  <div className="text-xs text-gray-400">
                    {anomaly.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}