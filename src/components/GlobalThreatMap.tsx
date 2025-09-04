import React, { useState, useEffect, useRef } from 'react';
import { Globe, Filter, RefreshCw, AlertTriangle, MapPin, Activity } from 'lucide-react';
import { threatMapService, GlobalThreat, ThreatStatistics } from '../services/threatMapService';

interface GlobalThreatMapProps {
  className?: string;
}

interface MapFilters {
  hours: number;
  minSeverity: number;
  country: string;
  threatType: string;
}

export function GlobalThreatMap({ className = '' }: GlobalThreatMapProps) {
  const [threats, setThreats] = useState<GlobalThreat[]>([]);
  const [statistics, setStatistics] = useState<ThreatStatistics | null>(null);
  const [filters, setFilters] = useState<MapFilters>({
    hours: 24,
    minSeverity: 1,
    country: '',
    threatType: ''
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    // Load initial data
    loadThreats();
    loadStatistics();

    // Set up WebSocket listeners
    const unsubscribeThreat = threatMapService.onThreatDetected((threat) => {
      setThreats(prev => [threat, ...prev.slice(0, 999)]); // Keep last 1000 threats
      animateNewThreat(threat);
    });

    const unsubscribeConnection = threatMapService.onConnectionChange((connected) => {
      setIsConnected(connected);
    });

    // Set up periodic refresh
    const refreshInterval = setInterval(() => {
      loadThreats();
      loadStatistics();
    }, 30000); // Refresh every 30 seconds

    return () => {
      unsubscribeThreat();
      unsubscribeConnection();
      clearInterval(refreshInterval);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useEffect(() => {
    // Reload threats when filters change
    loadThreats();
  }, [filters]);

  const loadThreats = async () => {
    try {
      setIsLoading(true);
      const response = await threatMapService.getGlobalThreats({
        hours: filters.hours,
        min_severity: filters.minSeverity,
        country: filters.country || undefined,
        limit: 1000
      });
      
      let filteredThreats = response.threats;
      
      // Apply client-side threat type filter
      if (filters.threatType) {
        filteredThreats = filteredThreats.filter(t => t.threat_type === filters.threatType);
      }
      
      setThreats(filteredThreats);
    } catch (error) {
      console.error('Failed to load threats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadStatistics = async () => {
    try {
      const stats = await threatMapService.getThreatStatistics();
      setStatistics(stats);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    }
  };

  const animateNewThreat = (threat: any) => {
    // Add visual animation for new threats
    if (mapRef.current) {
      const threatElement = document.createElement('div');
      threatElement.className = 'absolute w-4 h-4 bg-red-500 rounded-full animate-ping pointer-events-none';
      threatElement.style.left = `${Math.random() * 90 + 5}%`;
      threatElement.style.top = `${Math.random() * 80 + 10}%`;
      
      mapRef.current.appendChild(threatElement);
      
      setTimeout(() => {
        if (mapRef.current && threatElement.parentNode) {
          mapRef.current.removeChild(threatElement);
        }
      }, 3000);
    }
  };

  const handleFilterChange = (key: keyof MapFilters, value: string | number) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const getSeverityColor = (severity: number) => {
    return threatMapService.getSeverityColor(severity);
  };

  const getThreatTypeIcon = (threatType: string) => {
    return threatMapService.getThreatTypeIcon(threatType);
  };

  const getCountryThreats = () => {
    const countryMap = new Map<string, { count: number; maxSeverity: number; threats: GlobalThreat[] }>();
    
    threats.forEach(threat => {
      const countryCode = threat.source_location.country_code;
      if (!countryMap.has(countryCode)) {
        countryMap.set(countryCode, { count: 0, maxSeverity: 0, threats: [] });
      }
      
      const country = countryMap.get(countryCode)!;
      country.count++;
      country.maxSeverity = Math.max(country.maxSeverity, threat.severity);
      country.threats.push(threat);
    });
    
    return Array.from(countryMap.entries()).map(([code, data]) => ({
      countryCode: code,
      ...data
    })).sort((a, b) => b.count - a.count);
  };

  const uniqueThreatTypes = Array.from(new Set(threats.map(t => t.threat_type)));
  const uniqueCountries = Array.from(new Set(threats.map(t => t.source_location.country_code)));

  return (
    <div className={`bg-gray-800 rounded-xl border border-gray-700 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Globe className="h-6 w-6 text-blue-400 mr-2" />
            Global Threat Map
          </h2>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-300">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              <Filter className="h-4 w-4 text-gray-300" />
            </button>
            <button
              onClick={() => { loadThreats(); loadStatistics(); }}
              disabled={isLoading}
              className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 rounded-lg transition-colors"
            >
              <RefreshCw className={`h-4 w-4 text-white ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Statistics */}
        {statistics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-gray-900 rounded-lg p-3">
              <div className="text-red-400 text-lg font-bold">{statistics.overview.active_threats}</div>
              <div className="text-gray-400 text-xs">Active Threats</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3">
              <div className="text-orange-400 text-lg font-bold">{statistics.overview.threats_last_hour}</div>
              <div className="text-gray-400 text-xs">Last Hour</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3">
              <div className="text-blue-400 text-lg font-bold">{statistics.overview.countries_affected}</div>
              <div className="text-gray-400 text-xs">Countries</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3">
              <div className="text-purple-400 text-lg font-bold">{statistics.overview.avg_severity.toFixed(1)}</div>
              <div className="text-gray-400 text-xs">Avg Severity</div>
            </div>
          </div>
        )}

        {/* Filters */}
        {showFilters && (
          <div className="bg-gray-900 rounded-lg p-4 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Time Range</label>
                <select
                  value={filters.hours}
                  onChange={(e) => handleFilterChange('hours', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value={1}>Last Hour</option>
                  <option value={6}>Last 6 Hours</option>
                  <option value={24}>Last 24 Hours</option>
                  <option value={168}>Last Week</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Min Severity</label>
                <select
                  value={filters.minSeverity}
                  onChange={(e) => handleFilterChange('minSeverity', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value={1}>All (1+)</option>
                  <option value={3}>Medium (3+)</option>
                  <option value={6}>High (6+)</option>
                  <option value={8}>Critical (8+)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Country</label>
                <select
                  value={filters.country}
                  onChange={(e) => handleFilterChange('country', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="">All Countries</option>
                  {uniqueCountries.map(country => (
                    <option key={country} value={country}>{country}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Threat Type</label>
                <select
                  value={filters.threatType}
                  onChange={(e) => handleFilterChange('threatType', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="">All Types</option>
                  {uniqueThreatTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Map Visualization */}
      <div className="p-6">
        <div 
          ref={mapRef}
          className="relative bg-gray-900 rounded-lg h-96 overflow-hidden mb-6"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23374151' fill-opacity='0.1'%3E%3Ccircle cx='7' cy='7' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}
        >
          {/* Grid overlay */}
          <div className="absolute inset-0 opacity-20">
            <div className="grid grid-cols-12 grid-rows-8 h-full">
              {Array.from({ length: 96 }).map((_, i) => (
                <div key={i} className="border border-gray-600"></div>
              ))}
            </div>
          </div>

          {/* Threat indicators */}
          {threats.slice(0, 50).map((threat, index) => {
            const x = threat.source_location.longitude ? 
              ((threat.source_location.longitude + 180) / 360) * 100 : 
              Math.random() * 90 + 5;
            const y = threat.source_location.latitude ? 
              ((90 - threat.source_location.latitude) / 180) * 100 : 
              Math.random() * 80 + 10;

            return (
              <div
                key={threat.id}
                className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                style={{
                  left: `${Math.max(2, Math.min(98, x))}%`,
                  top: `${Math.max(2, Math.min(98, y))}%`,
                  animationDelay: `${index * 0.1}s`
                }}
                title={threatMapService.formatThreatDescription(threat)}
              >
                <div
                  className="w-3 h-3 rounded-full animate-pulse"
                  style={{ backgroundColor: getSeverityColor(threat.severity) }}
                >
                  <div
                    className="absolute inset-0 rounded-full animate-ping"
                    style={{ backgroundColor: getSeverityColor(threat.severity) }}
                  ></div>
                </div>
                
                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                  {getThreatTypeIcon(threat.threat_type)} {threat.source_location.city}, {threat.source_location.country}
                  <br />
                  Severity: {threat.severity}/10
                </div>
              </div>
            );
          })}

          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-gray-900/50 flex items-center justify-center">
              <div className="flex items-center space-x-2 text-white">
                <Activity className="h-5 w-5 animate-spin" />
                <span>Loading threats...</span>
              </div>
            </div>
          )}
        </div>

        {/* Country Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Top Countries */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              <MapPin className="h-5 w-5 text-red-400 mr-2" />
              Top Threat Sources
            </h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {getCountryThreats().slice(0, 10).map(({ countryCode, count, maxSeverity, threats }) => (
                <div key={countryCode} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getSeverityColor(maxSeverity) }}
                    ></div>
                    <span className="text-white text-sm">
                      {threats[0]?.source_location.country || countryCode}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-red-400 font-medium">{count}</div>
                    <div className="text-gray-400 text-xs">threats</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Threat Types */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              <AlertTriangle className="h-5 w-5 text-orange-400 mr-2" />
              Threat Types
            </h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {statistics?.threat_types.slice(0, 10).map(({ type, count }) => (
                <div key={type} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getThreatTypeIcon(type)}</span>
                    <span className="text-white text-sm capitalize">
                      {type.replace('_', ' ')}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-orange-400 font-medium">{count}</div>
                    <div className="text-gray-400 text-xs">events</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}