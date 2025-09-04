import React from 'react';
import { Shield, Activity, AlertTriangle, Network, Database } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { ThreatMap } from './ThreatMap';
import { IncidentList } from './IncidentList';
import { NetworkMonitor } from './NetworkMonitor';
import { SystemStatus } from './SystemStatus';
import { AlertPanel } from './AlertPanel';
import { ThreatDetectionPanel } from './ThreatDetectionPanel';
import { SystemAlertPanel } from './SystemAlertPanel';
import { UserManagement } from './UserManagement';
import { CriticalIncidentsPanel } from './CriticalIncidentsPanel';
import { useIncidentData } from '../hooks/useIncidentData';
import { User } from '../types/user';

export function Dashboard({ user, onLogout }: DashboardProps) {
  const { 
    incidents, 
    networkTraffic, 
    systemStatus, 
    alerts, 
    threatDetections, 
    anomalies, 
    isMonitoring, 
    toggleMonitoring,
    backendConnected,
    backendStats,
    threatBackendConnected,
    criticalIncidents,
    criticalIncidentsConnected
  } = useIncidentData();
  
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [activeSection, setActiveSection] = React.useState('overview');

  // Safe array defaults and calculations
  const safeIncidents = incidents || [];
  const safeSystemStatus = systemStatus || [];
  const safeAlerts = alerts || [];
  const safeThreatDetections = threatDetections || [];
  const safeAnomalies = anomalies || [];
  const safeNetworkTraffic = networkTraffic || [];

  const criticalIncidentsCount = criticalIncidents?.length || 0;
  const localCriticalIncidents = safeIncidents.filter(i => i.severity === 'critical').length;
  const highIncidents = safeIncidents.filter(i => i.severity === 'high').length;
  const activeLocalIncidents = safeIncidents.filter(i => i.status !== 'resolved').length;
  const onlineSystemsCount = safeSystemStatus.filter(s => s.status === 'online').length;
  const unacknowledgedAlerts = safeAlerts.filter(a => !a.acknowledged).length;
  const highConfidenceThreats = safeThreatDetections.filter(t => t.confidence > 80).length;
  const criticalAnomalies = safeAnomalies.filter(a => a.severity === 'critical').length;

  // Calculate total active threats from all sources
  const totalActiveThreats = activeLocalIncidents + highConfidenceThreats + criticalIncidentsCount + (backendStats?.active_threats || 0);

  const stats = [
    {
      title: 'Active Threats',
      value: totalActiveThreats,
      icon: Shield,
      color: 'text-red-400',
      bgColor: 'bg-red-500/20'
    },
    {
      title: 'Critical Incidents',
      value: criticalIncidentsCount + localCriticalIncidents + criticalAnomalies,
      icon: AlertTriangle,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20'
    },
    {
      title: 'Systems Online',
      value: `${onlineSystemsCount}/${safeSystemStatus.length}`,
      icon: Database,
      color: 'text-green-400',
      bgColor: 'bg-green-500/20'
    },
    {
      title: 'Network Traffic',
      value: `${safeNetworkTraffic.length}`,
      icon: Network,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20'
    }
  ];

  const sidebarStats = {
    activeThreats: totalActiveThreats,
    networkTraffic: safeNetworkTraffic.length,
    systemsOnline: `${onlineSystemsCount}/${safeSystemStatus.length}`,
    activeAlerts: safeAlerts.filter(a => !a.acknowledged && !a.isDuplicate).length,
    recentIncidents: safeIncidents.length,
    criticalIncidents: criticalIncidentsCount + localCriticalIncidents
  };

  const getSectionTitle = () => {
    switch (activeSection) {
      case 'threats':
        return 'Active Threats';
      case 'critical':
        return 'Critical Incidents';
      case 'network':
        return 'Network Traffic Monitor';
      case 'systems':
        return 'Systems Status';
      case 'alerts':
        return 'Active Alerts';
      case 'incidents':
        return 'Recent Incidents';
      case 'detection':
        return 'Threat Detection & Anomalies';
      case 'user-management':
        return 'User Management';
      case 'critical-incidents':
        return 'Critical Incidents';
      default:
        return 'Dashboard Overview';
    }
  };

  const renderContent = () => {
    switch (activeSection) {
      case 'threats':
        return (
          <div className="space-y-6">
            {/* Active Threats Summary */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Active Threats Overview</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-red-900/30 rounded-lg p-4">
                  <div className="text-red-400 text-2xl font-bold">{activeLocalIncidents}</div>
                  <div className="text-gray-400 text-sm">Local Incidents</div>
                </div>
                <div className="bg-orange-900/30 rounded-lg p-4">
                  <div className="text-orange-400 text-2xl font-bold">{highConfidenceThreats}</div>
                  <div className="text-gray-400 text-sm">High Confidence</div>
                </div>
                <div className="bg-purple-900/30 rounded-lg p-4">
                  <div className="text-purple-400 text-2xl font-bold">{criticalIncidentsCount}</div>
                  <div className="text-gray-400 text-sm">Critical Backend</div>
                </div>
                <div className="bg-blue-900/30 rounded-lg p-4">
                  <div className="text-blue-400 text-2xl font-bold">{backendStats?.active_threats || 0}</div>
                  <div className="text-gray-400 text-sm">Backend Threats</div>
                </div>
              </div>
              {backendConnected && backendStats && (
                <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-blue-300 mb-2">Backend Statistics</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                    <div>
                      <span className="text-gray-400">Total Threats:</span>
                      <span className="text-white ml-1">{backendStats.total_threats}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Blocked IPs:</span>
                      <span className="text-white ml-1">{backendStats.blocked_ips}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Last Hour:</span>
                      <span className="text-white ml-1">{backendStats.threats_last_hour}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Last 24h:</span>
                      <span className="text-white ml-1">{backendStats.threats_last_24h}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <ThreatMap incidents={safeIncidents} />
            <IncidentList incidents={safeIncidents.filter(i => i.status !== 'resolved')} />
          </div>
        );
      case 'critical':
        return (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Critical Incidents Overview</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-red-900/30 rounded-lg p-4">
                  <div className="text-red-400 text-2xl font-bold">{criticalIncidentsCount + localCriticalIncidents}</div>
                  <div className="text-gray-400 text-sm">Critical Incidents</div>
                </div>
                <div className="bg-orange-900/30 rounded-lg p-4">
                  <div className="text-orange-400 text-2xl font-bold">{highIncidents}</div>
                  <div className="text-gray-400 text-sm">High Priority</div>
                </div>
                <div className="bg-yellow-900/30 rounded-lg p-4">
                  <div className="text-yellow-400 text-2xl font-bold">{activeLocalIncidents}</div>
                  <div className="text-gray-400 text-sm">Total Active</div>
                </div>
              </div>
            </div>
            <IncidentList incidents={safeIncidents.filter(i => i.severity === 'critical' || i.severity === 'high')} />
          </div>
        );
      case 'network':
        return (
          <div className="space-y-6">
            <NetworkMonitor networkTraffic={safeNetworkTraffic} />
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Network Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-900 rounded-lg p-4">
                  <div className="text-blue-400 text-xl font-bold">
                    {safeNetworkTraffic.filter(t => t.suspicious).length}
                  </div>
                  <div className="text-gray-400 text-sm">Suspicious Connections</div>
                </div>
                <div className="bg-gray-900 rounded-lg p-4">
                  <div className="text-green-400 text-xl font-bold">
                    {Math.round((safeNetworkTraffic.reduce((sum, t) => sum + t.bytes, 0) / 1024 / 1024) * 100) / 100}MB
                  </div>
                  <div className="text-gray-400 text-sm">Total Data Volume</div>
                </div>
              </div>
            </div>
          </div>
        );
      case 'systems':
        return (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">System Health Overview</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-green-900/30 rounded-lg p-4">
                  <div className="text-green-400 text-xl font-bold">
                    {safeSystemStatus.filter(s => s.status === 'online').length}
                  </div>
                  <div className="text-gray-400 text-sm">Online</div>
                </div>
                <div className="bg-yellow-900/30 rounded-lg p-4">
                  <div className="text-yellow-400 text-xl font-bold">
                    {safeSystemStatus.filter(s => s.status === 'warning').length}
                  </div>
                  <div className="text-gray-400 text-sm">Warning</div>
                </div>
                <div className="bg-red-900/30 rounded-lg p-4">
                  <div className="text-red-400 text-xl font-bold">
                    {safeSystemStatus.filter(s => s.status === 'error').length}
                  </div>
                  <div className="text-gray-400 text-sm">Error</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-gray-400 text-xl font-bold">
                    {safeSystemStatus.filter(s => s.status === 'offline').length}
                  </div>
                  <div className="text-gray-400 text-sm">Offline</div>
                </div>
              </div>
            </div>
            <SystemStatus systems={safeSystemStatus} />
          </div>
        );
      case 'alerts':
        return (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Alert Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-red-900/30 rounded-lg p-4">
                  <div className="text-red-400 text-xl font-bold">
                    {safeAlerts.filter(a => a.type === 'critical').length}
                  </div>
                  <div className="text-gray-400 text-sm">Critical</div>
                </div>
                <div className="bg-orange-900/30 rounded-lg p-4">
                  <div className="text-orange-400 text-xl font-bold">
                    {safeAlerts.filter(a => a.type === 'error').length}
                  </div>
                  <div className="text-gray-400 text-sm">Error</div>
                </div>
                <div className="bg-yellow-900/30 rounded-lg p-4">
                  <div className="text-yellow-400 text-xl font-bold">
                    {safeAlerts.filter(a => a.type === 'warning').length}
                  </div>
                  <div className="text-gray-400 text-sm">Warning</div>
                </div>
                <div className="bg-blue-900/30 rounded-lg p-4">
                  <div className="text-blue-400 text-xl font-bold">
                    {safeAlerts.filter(a => !a.acknowledged).length}
                  </div>
                  <div className="text-gray-400 text-sm">Unacknowledged</div>
                </div>
              </div>
            </div>
            <AlertPanel alerts={safeAlerts} />
          </div>
        );
      case 'incidents':
        return (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Incident Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-red-900/30 rounded-lg p-4">
                  <div className="text-red-400 text-xl font-bold">{criticalIncidentsCount}</div>
                  <div className="text-gray-400 text-sm">Critical</div>
                </div>
                <div className="bg-orange-900/30 rounded-lg p-4">
                  <div className="text-orange-400 text-xl font-bold">{highIncidents}</div>
                  <div className="text-gray-400 text-sm">High</div>
                </div>
                <div className="bg-yellow-900/30 rounded-lg p-4">
                  <div className="text-yellow-400 text-xl font-bold">
                    {safeIncidents.filter(i => i.severity === 'medium').length}
                  </div>
                  <div className="text-gray-400 text-sm">Medium</div>
                </div>
                <div className="bg-green-900/30 rounded-lg p-4">
                  <div className="text-green-400 text-xl font-bold">
                    {safeIncidents.filter(i => i.status === 'resolved').length}
                  </div>
                  <div className="text-gray-400 text-sm">Resolved</div>
                </div>
              </div>
            </div>
            <IncidentList incidents={safeIncidents} />
          </div>
        );
      case 'detection':
        return <ThreatDetectionPanel threatDetections={safeThreatDetections} anomalies={safeAnomalies} />;
      case 'email-alerts':
        return <SystemAlertPanel />;
      case 'user-management':
        return user ? <UserManagement currentUser={user} /> : null;
      case 'critical-incidents':
        return <CriticalIncidentsPanel />;
      default:
        return (
          <>
            <div className="space-y-6">
              {/* Welcome Section */}
              <div className="text-center py-8">
                <Shield className="h-16 w-16 text-blue-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">
                  Cybersecurity Operations Center
                </h2>
                <p className="text-gray-400">
                  Real-time threat monitoring and incident response
                </p>
              </div>

              {/* Stats Overview */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {stats.map((stat, index) => (
                  <div key={index} className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-gray-400 text-sm font-medium">{stat.title}</p>
                        <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
                      </div>
                      <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                        <stat.icon className={`h-6 w-6 ${stat.color}`} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Advanced Detection Summary */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Shield className="h-5 w-5 text-blue-400 mr-2" />
                  Advanced Detection Summary
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-900 rounded-lg p-4">
                    <div className="text-red-400 text-xl font-bold">{highConfidenceThreats}</div>
                    <div className="text-gray-400 text-sm">High Confidence Threats</div>
                  </div>
                  <div className="bg-gray-900 rounded-lg p-4">
                    <div className="text-purple-400 text-xl font-bold">{safeAnomalies.length}</div>
                    <div className="text-gray-400 text-sm">Anomalies Detected</div>
                  </div>
                  <div className="bg-gray-900 rounded-lg p-4">
                    <div className="text-blue-400 text-xl font-bold">
                      {safeAlerts.filter(a => a.correlationId).length}
                    </div>
                    <div className="text-gray-400 text-sm">Correlated Alerts</div>
                  </div>
                  <div className="bg-gray-900 rounded-lg p-4">
                    <div className="text-green-400 text-xl font-bold">
                      {safeAlerts.filter(a => a.isDuplicate).length}
                    </div>
                    <div className="text-gray-400 text-sm">Deduplicated</div>
                  </div>
                </div>
              </div>

              {/* Global Threat Map */}
              <ThreatMap incidents={safeIncidents} />

              {/* Navigation Hint */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 text-center">
                <p className="text-gray-400">
                  Use the sidebar navigation to access detailed monitoring sections including advanced threat detection
                </p>
              </div>
            </div>
          </>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        stats={sidebarStats}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold text-white">{getSectionTitle()}</h1>
                <p className="text-sm text-gray-400">
                  Security Operations Center{user ? ` • Welcome, ${user.firstName} ${user.lastName}` : ''}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {user && onLogout && (
                <button
                  onClick={onLogout}
                  className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white rounded-lg transition-colors"
                >
                  Logout
                </button>
              )}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-sm text-gray-300">
                  {isMonitoring ? 'Monitoring Active' : 'Monitoring Paused'}
                </span>
                <div className={`w-2 h-2 rounded-full ml-2 ${backendConnected ? 'bg-blue-400' : 'bg-gray-400'}`} />
                <span className="text-xs text-gray-400">
                  Backend {backendConnected ? 'Connected' : 'Disconnected'}
                </span>
                <div className={`w-2 h-2 rounded-full ml-2 ${threatBackendConnected ? 'bg-green-400' : 'bg-gray-400'}`} />
                <span className="text-xs text-gray-400">
                  Threats {threatBackendConnected ? 'Connected' : 'Disconnected'}
                </span>
                <div className={`w-2 h-2 rounded-full ml-2 ${criticalIncidentsConnected ? 'bg-red-400' : 'bg-gray-400'}`} />
                <span className="text-xs text-gray-400">
                  Critical {criticalIncidentsConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <button
                onClick={toggleMonitoring}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isMonitoring 
                    ? 'bg-red-600 hover:bg-red-700 text-white' 
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                {isMonitoring ? 'Pause' : 'Resume'} Monitoring
              </button>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 px-6 py-6 overflow-auto">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}