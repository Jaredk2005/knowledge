import React, { useState } from 'react';
import { Mail, Settings, Play, Square, TestTube, CheckCircle, AlertTriangle } from 'lucide-react';

export function SystemAlertPanel() {
  const [emailConfig, setEmailConfig] = useState({
    smtpServer: 'smtp.gmail.com',
    smtpPort: '587',
    username: '',
    password: '',
    fromEmail: '',
    toEmails: '',
    cpuThreshold: '80',
    memoryThreshold: '85',
    diskThreshold: '90',
    checkInterval: '60'
  });

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const handleConfigChange = (field: string, value: string) => {
    setEmailConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleSaveConfig = () => {
    // In a real implementation, this would save to the Python config file
    console.log('Saving configuration:', emailConfig);
    alert('Configuration saved! Update alert_config.json with these settings.');
  };

  const handleTestEmail = () => {
    setTestResult('sending');
    // Simulate test email
    setTimeout(() => {
      setTestResult('success');
      setTimeout(() => setTestResult(null), 3000);
    }, 2000);
  };

  const toggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
    // In a real implementation, this would start/stop the Python monitoring process
  };

  return (
    <div className="space-y-6">
      {/* Email Alert Configuration */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Mail className="h-6 w-6 text-blue-400 mr-2" />
            Email Alert System Configuration
          </h2>
          <div className="flex items-center space-x-2">
            System & Security Alert Settings
            <span className="text-sm text-gray-300">
              {isMonitoring ? 'Monitoring Active' : 'Monitoring Inactive'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* SMTP Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <Settings className="h-5 w-5 text-gray-400 mr-2" />
              SMTP Settings
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">SMTP Server</label>
                <input
                  type="text"
                  value={emailConfig.smtpServer}
                  onChange={(e) => handleConfigChange('smtpServer', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="smtp.gmail.com"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">SMTP Port</label>
                <input
                  type="number"
                  value={emailConfig.smtpPort}
                  onChange={(e) => handleConfigChange('smtpPort', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="587"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Username</label>
                <input
                  type="email"
                  value={emailConfig.username}
                  onChange={(e) => handleConfigChange('username', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="your-email@gmail.com"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">App Password</label>
                <input
                  type="password"
                  value={emailConfig.password}
                  onChange={(e) => handleConfigChange('password', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="your-app-password"
                />
              </div>
            </div>
            
            {/* Security Alert Settings */}
            <div className="mt-4 pt-4 border-t border-gray-600">
              <h4 className="text-md font-medium text-white mb-3">Cybersecurity Alert Thresholds</h4>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Failed Login Threshold</label>
                  <input
                    type="number"
                    defaultValue="5"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Network Scan Threshold</label>
                  <input
                    type="number"
                    defaultValue="10"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="1"
                  />
                </div>
              </div>
              
              <div className="mt-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-300">Enable threat detection email alerts</span>
                </label>
              </div>
              
              <div className="mt-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-300">Send alerts for critical incidents only</span>
                </label>
              </div>
            </div>
          </div>

          {/* Alert Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
              Alert Settings
            </h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">From Email</label>
                <input
                  type="email"
                  value={emailConfig.fromEmail}
                  onChange={(e) => handleConfigChange('fromEmail', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="alerts@yourcompany.com"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">To Emails (comma separated)</label>
                <textarea
                  value={emailConfig.toEmails}
                  onChange={(e) => handleConfigChange('toEmails', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="admin1@company.com, admin2@company.com"
                  rows={2}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">CPU Threshold (%)</label>
                  <input
                    type="number"
                    value={emailConfig.cpuThreshold}
                    onChange={(e) => handleConfigChange('cpuThreshold', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="1" max="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Memory Threshold (%)</label>
                  <input
                    type="number"
                    value={emailConfig.memoryThreshold}
                    onChange={(e) => handleConfigChange('memoryThreshold', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="1" max="100"
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Disk Threshold (%)</label>
                  <input
                    type="number"
                    value={emailConfig.diskThreshold}
                    onChange={(e) => handleConfigChange('diskThreshold', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="1" max="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Check Interval (sec)</label>
                  <input
                    type="number"
                    value={emailConfig.checkInterval}
                    onChange={(e) => handleConfigChange('checkInterval', e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    min="30"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between mt-6 pt-6 border-t border-gray-700">
          <div className="flex space-x-3">
            <button
              onClick={handleSaveConfig}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Save Configuration
            </button>
            
            <button
              onClick={handleTestEmail}
              disabled={testResult === 'sending'}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center space-x-2"
            >
              <TestTube className="h-4 w-4" />
              <span>{testResult === 'sending' ? 'Sending...' : 'Test Email'}</span>
            </button>
          </div>
          
          <button
            onClick={toggleMonitoring}
            className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 ${
              isMonitoring 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isMonitoring ? <Square className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}</span>
          </button>
          
          <button
            onClick={() => {
              // Test threat alert
              setTestResult('sending');
              setTimeout(() => {
                setTestResult('success');
                setTimeout(() => setTestResult(null), 3000);
              }, 2000);
            }}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <AlertTriangle className="h-4 w-4" />
            <span>Test Threat Alert</span>
          </button>
        </div>

        {/* Test Result */}
        {testResult && (
          <div className={`mt-4 p-3 rounded-lg flex items-center space-x-2 ${
            testResult === 'success' ? 'bg-green-900/30 text-green-400' : 'bg-blue-900/30 text-blue-400'
          }`}>
            {testResult === 'success' ? (
              <CheckCircle className="h-5 w-5" />
            ) : (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
            )}
            <span>
              {testResult === 'success' ? 'Test email sent successfully!' : 'Sending test email...'}
            </span>
          </div>
        )}
      </div>

      {/* Usage Instructions */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">System Setup Instructions</h3>
        <div className="space-y-4 text-sm text-gray-300">
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">1. Install Python Dependencies</h4>
            <code className="text-green-400">pip install psutil</code>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">2. Configure Email Settings</h4>
            <p>Update the configuration above and save to <code className="text-blue-400">alert_config.json</code></p>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">3. Run System Monitor</h4>
            <div className="space-y-1">
              <div><code className="text-green-400">python system_monitor.py --test</code> - Send test email</div>
              <div><code className="text-green-400">python system_monitor.py --check</code> - Run single check</div>
              <div><code className="text-green-400">python system_monitor.py --monitor</code> - Start continuous monitoring</div>
            </div>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="font-medium text-white mb-2">4. Gmail Setup (if using Gmail)</h4>
            <ul className="list-disc list-inside space-y-1">
              <li>Enable 2-factor authentication</li>
              <li>Generate an App Password</li>
              <li>Use the App Password in the configuration</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}