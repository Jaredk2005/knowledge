import React from 'react';
import { 
  Shield, 
  AlertTriangle, 
  Activity, 
  Server, 
  Bell, 
  FileText, 
  ChevronLeft, 
  ChevronRight,
  BarChart3,
  Brain,
  Mail,
  UserPlus,
  Lock
} from 'lucide-react';

interface SidebarProps {
  isCollapsed: boolean;
  onToggle: () => void;
  activeSection: string;
  onSectionChange: (section: string) => void;
  stats: {
    activeThreats: number;
    criticalIncidents: number;
    networkTraffic: number;
    systemsOnline: string;
    activeAlerts: number;
    recentIncidents: number;
    criticalIncidents: number;
  };
}

export function Sidebar({ 
  isCollapsed, 
  onToggle, 
  activeSection, 
  onSectionChange, 
  stats 
}: SidebarProps) {
  const menuItems = [
    {
      id: 'overview',
      label: 'Dashboard Overview',
      icon: BarChart3,
      count: null
    },
    {
      id: 'threats',
      label: 'Active Threats',
      icon: Shield,
      count: stats.activeThreats,
      color: 'text-red-400'
    },
    {
      id: 'critical',
      label: 'Critical Incidents',
      icon: AlertTriangle,
      count: stats.criticalIncidents,
      color: 'text-orange-400'
    },
    {
      id: 'network',
      label: 'Network Monitor',
      icon: Activity,
      count: stats.networkTraffic,
      color: 'text-green-400'
    },
    {
      id: 'systems',
      label: 'Systems Status',
      icon: Server,
      count: stats.systemsOnline,
      color: 'text-blue-400'
    },
    {
      id: 'alerts',
      label: 'Active Alerts',
      icon: Bell,
      count: stats.activeAlerts,
      color: 'text-yellow-400'
    },
    {
      id: 'incidents',
      label: 'Recent Incidents',
      icon: FileText,
      count: stats.recentIncidents,
      color: 'text-purple-400'
    },
    {
      id: 'detection',
      label: 'Threat Detection',
      icon: Brain,
      count: null,
      color: 'text-cyan-400'
    },
    {
      id: 'email-alerts',
      label: 'Email Alert System',
      icon: Mail,
      count: null,
      color: 'text-indigo-400'
    },
    {
      id: 'critical-incidents',
      label: 'Critical Incidents',
      icon: AlertTriangle,
      count: stats.criticalIncidents,
      color: 'text-red-500'
    },
    {
      id: 'user-management',
      label: 'User Management',
      icon: UserPlus,
      count: null,
      color: 'text-purple-400'
    }
  ];

  return (
    <div className={`bg-gray-800 border-r border-gray-700 transition-all duration-300 ${
      isCollapsed ? 'w-16' : 'w-64'
    } flex flex-col`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex items-center space-x-2">
              <Shield className="h-6 w-6 text-blue-400" />
              <span className="font-semibold text-white">SOC Menu</span>
            </div>
          )}
          <button
            onClick={onToggle}
            className="p-1 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
          >
            {isCollapsed ? (
              <ChevronRight className="h-5 w-5" />
            ) : (
              <ChevronLeft className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 p-2">
        <div className="space-y-1">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-3 rounded-lg transition-colors ${
                activeSection === item.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
              title={isCollapsed ? item.label : undefined}
            >
              <item.icon className={`h-5 w-5 ${
                activeSection === item.id ? 'text-white' : item.color || 'text-gray-400'
              }`} />
              
              {!isCollapsed && (
                <>
                  <span className="flex-1 text-left text-sm font-medium">
                    {item.label}
                  </span>
                  {item.count !== null && (
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      activeSection === item.id
                        ? 'bg-white/20 text-white'
                        : 'bg-gray-700 text-gray-300'
                    }`}>
                      {item.count}
                    </span>
                  )}
                </>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Status Indicator */}
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          {!isCollapsed && (
            <span className="text-xs text-gray-400">System Active</span>
          )}
        </div>
      </div>
    </div>
  );
}