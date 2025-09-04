import React, { useMemo } from 'react';
import { Activity, TrendingUp, AlertTriangle } from 'lucide-react';
import { NetworkTraffic } from '../types/incident';

interface NetworkMonitorProps {
  networkTraffic: NetworkTraffic[];
}

export function NetworkMonitor({ networkTraffic }: NetworkMonitorProps) {
  const trafficStats = useMemo(() => {
    const total = networkTraffic.length;
    const suspicious = networkTraffic.filter(t => t.suspicious).length;
    const totalBytes = networkTraffic.reduce((sum, t) => sum + t.bytes, 0);
    
    return {
      total,
      suspicious,
      totalBytes,
      suspiciousPercentage: total > 0 ? Math.round((suspicious / total) * 100) : 0
    };
  }, [networkTraffic]);

  const recentTraffic = networkTraffic.slice(0, 10);

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center">
          <Activity className="h-6 w-6 text-green-400 mr-2" />
          Network Traffic Monitor
        </h2>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-green-400 text-sm">Live</span>
        </div>
      </div>

      {/* Traffic Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-blue-400 text-2xl font-bold">{trafficStats.total}</div>
              <div className="text-gray-400 text-sm">Total Packets</div>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-red-400 text-2xl font-bold">{trafficStats.suspicious}</div>
              <div className="text-gray-400 text-sm">Suspicious</div>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-400" />
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-green-400 text-2xl font-bold">
                {(trafficStats.totalBytes / 1024).toFixed(1)}KB
              </div>
              <div className="text-gray-400 text-sm">Data Volume</div>
            </div>
            <Activity className="h-8 w-8 text-green-400" />
          </div>
        </div>
      </div>

      {/* Recent Traffic */}
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-white mb-3">Recent Network Activity</h3>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {recentTraffic.map((traffic, index) => (
            <div
              key={index}
              className={`flex items-center justify-between p-3 rounded-lg ${
                traffic.suspicious ? 'bg-red-900/30 border border-red-700/50' : 'bg-gray-900'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  traffic.suspicious ? 'bg-red-400' : 'bg-green-400'
                }`}></div>
                <div>
                  <div className="text-white text-sm font-medium">
                    {traffic.source} → {traffic.destination}
                  </div>
                  <div className="text-gray-400 text-xs">
                    {traffic.protocol} • {traffic.bytes} bytes
                  </div>
                </div>
              </div>
              <div className="text-gray-400 text-xs">
                {traffic.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}