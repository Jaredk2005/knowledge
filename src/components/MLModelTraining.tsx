import React, { useState, useEffect } from 'react';
import { Brain, Play, Square, BarChart3, Settings, Download, Upload, RefreshCw, CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_score: number;
  training_samples: number;
  test_samples: number;
  training_time: number;
  last_trained: string;
  feature_importance: Record<string, number>;
}

interface ModelInfo {
  model_status: string;
  feature_count: number;
  threat_categories: string[];
  last_trained?: string;
  accuracy?: number;
  f1_score?: number;
}

interface TrainingConfig {
  use_database: boolean;
  augment_data: boolean;
  training_samples?: number;
}

export function MLModelTraining() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    use_database: true,
    augment_data: true,
    training_samples: 5000
  });
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null);
  const [modelReport, setModelReport] = useState<string>('');

  const mlApiUrl = 'http://localhost:8003/api/ml';

  useEffect(() => {
    checkMLApiConnection();
    loadModelInfo();
    loadModelMetrics();
  }, []);

  const checkMLApiConnection = async () => {
    try {
      const response = await fetch(`${mlApiUrl}/health`);
      if (response.ok) {
        setIsConnected(true);
        addToLog('✅ Connected to ML API server');
      } else {
        setIsConnected(false);
        addToLog('❌ ML API server not responding');
      }
    } catch (error) {
      setIsConnected(false);
      addToLog('❌ Failed to connect to ML API server');
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await fetch(`${mlApiUrl}/model/info`);
      if (response.ok) {
        const info = await response.json();
        setModelInfo(info);
        addToLog(`📊 Model loaded: ${info.model_status}`);
      }
    } catch (error) {
      console.error('Failed to load model info:', error);
      addToLog('❌ Failed to load model information');
    }
  };

  const loadModelMetrics = async () => {
    try {
      const response = await fetch(`${mlApiUrl}/model/metrics`);
      if (response.ok) {
        const metrics = await response.json();
        setModelMetrics(metrics);
        addToLog(`📈 Model metrics loaded - Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
      }
    } catch (error) {
      console.error('Failed to load model metrics:', error);
      // Don't log error if model isn't trained yet
    }
  };

  const addToLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setTrainingLog(prev => [`[${timestamp}] ${message}`, ...prev.slice(0, 49)]);
  };

  const startTraining = async () => {
    if (!isConnected) {
      addToLog('❌ Cannot start training - ML API not connected');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    addToLog('🚀 Starting model training...');

    try {
      const response = await fetch(`${mlApiUrl}/model/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingConfig),
      });

      if (response.ok) {
        const result = await response.json();
        addToLog(`✅ ${result.message}`);
        
        // Simulate training progress
        simulateTrainingProgress();
        
        // Poll for completion
        pollTrainingStatus();
      } else {
        const error = await response.text();
        addToLog(`❌ Training failed: ${error}`);
        setIsTraining(false);
      }
    } catch (error) {
      addToLog(`❌ Training error: ${error}`);
      setIsTraining(false);
    }
  };

  const simulateTrainingProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 10;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
      }
      setTrainingProgress(progress);
    }, 1000);
  };

  const pollTrainingStatus = () => {
    const pollInterval = setInterval(async () => {
      try {
        await loadModelInfo();
        await loadModelMetrics();
        
        // Check if training completed by looking for updated metrics
        if (modelMetrics && new Date(modelMetrics.last_trained) > new Date(Date.now() - 60000)) {
          addToLog('✅ Model training completed successfully!');
          setIsTraining(false);
          setTrainingProgress(100);
          clearInterval(pollInterval);
        }
      } catch (error) {
        // Continue polling
      }
    }, 5000);

    // Stop polling after 10 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      if (isTraining) {
        addToLog('⏰ Training timeout - check ML API logs');
        setIsTraining(false);
      }
    }, 600000);
  };

  const runBenchmark = async () => {
    try {
      addToLog('⚡ Running performance benchmark...');
      const response = await fetch(`${mlApiUrl}/model/benchmark`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const results = await response.json();
        setBenchmarkResults(results);
        addToLog(`📊 Benchmark completed: ${results.predictions_per_second.toFixed(1)} predictions/sec`);
      }
    } catch (error) {
      addToLog(`❌ Benchmark failed: ${error}`);
    }
  };

  const generateReport = async () => {
    try {
      addToLog('📄 Generating model report...');
      const response = await fetch(`${mlApiUrl}/model/report`);
      
      if (response.ok) {
        const result = await response.json();
        setModelReport(result.report);
        addToLog('✅ Model report generated');
      }
    } catch (error) {
      addToLog(`❌ Report generation failed: ${error}`);
    }
  };

  const testPrediction = async () => {
    try {
      addToLog('🧪 Testing model prediction...');
      const testEvent = {
        source_ip: '192.168.1.100',
        destination_ip: '10.0.0.5',
        packet_size: 1024,
        protocol: 'TCP',
        payload: 'test payload data'
      };

      const response = await fetch(`${mlApiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testEvent),
      });

      if (response.ok) {
        const result = await response.json();
        addToLog(`🎯 Test prediction: ${result.threat_type} (${(result.confidence * 100).toFixed(1)}% confidence)`);
      }
    } catch (error) {
      addToLog(`❌ Test prediction failed: ${error}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Brain className="h-6 w-6 text-purple-400 mr-2" />
            ML Threat Detection Model Training
          </h2>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-300">
                ML API {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button
              onClick={checkMLApiConnection}
              className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              <RefreshCw className="h-4 w-4 text-white" />
            </button>
          </div>
        </div>

        {/* Model Status */}
        {modelInfo && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-purple-400 text-lg font-bold">
                {modelInfo.model_status === 'trained' ? 'Trained' : 'Not Trained'}
              </div>
              <div className="text-gray-400 text-sm">Model Status</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-blue-400 text-lg font-bold">{modelInfo.feature_count}</div>
              <div className="text-gray-400 text-sm">Features</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-green-400 text-lg font-bold">{modelInfo.threat_categories.length}</div>
              <div className="text-gray-400 text-sm">Threat Types</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-orange-400 text-lg font-bold">
                {modelInfo.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
          </div>
        )}

        {/* Connection Warning */}
        {!isConnected && (
          <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-red-400" />
              <div>
                <p className="text-red-300 font-medium">ML API Server Not Connected</p>
                <p className="text-red-400 text-sm">Start the ML API server: <code>python ml_models/model_api.py</code></p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Training Configuration */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Settings className="h-5 w-5 text-gray-400 mr-2" />
          Training Configuration
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Data Source</label>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={trainingConfig.use_database}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, use_database: e.target.checked }))}
                  className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-300">Use database data</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={trainingConfig.augment_data}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, augment_data: e.target.checked }))}
                  className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-300">Augment training data</span>
              </label>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Training Samples</label>
            <input
              type="number"
              value={trainingConfig.training_samples || 5000}
              onChange={(e) => setTrainingConfig(prev => ({ ...prev, training_samples: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              min="1000"
              max="50000"
              step="1000"
            />
            <p className="text-xs text-gray-400 mt-1">Minimum 1,000 samples recommended</p>
          </div>
          
          <div className="flex items-end">
            <button
              onClick={startTraining}
              disabled={!isConnected || isTraining}
              className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
            >
              {isTraining ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Training...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Start Training</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="mt-6 p-4 bg-purple-900/20 border border-purple-700/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-purple-300 font-medium">Training Progress</span>
              <span className="text-purple-300">{trainingProgress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Model Performance */}
      {modelMetrics && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 text-blue-400 mr-2" />
            Model Performance Metrics
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-green-400 text-xl font-bold">{(modelMetrics.accuracy * 100).toFixed(1)}%</div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-blue-400 text-xl font-bold">{(modelMetrics.precision * 100).toFixed(1)}%</div>
              <div className="text-gray-400 text-sm">Precision</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-purple-400 text-xl font-bold">{(modelMetrics.recall * 100).toFixed(1)}%</div>
              <div className="text-gray-400 text-sm">Recall</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="text-orange-400 text-xl font-bold">{(modelMetrics.f1_score * 100).toFixed(1)}%</div>
              <div className="text-gray-400 text-sm">F1 Score</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Training Info */}
            <div className="bg-gray-900 rounded-lg p-4">
              <h4 className="text-white font-medium mb-3">Training Information</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Training Samples:</span>
                  <span className="text-white">{modelMetrics.training_samples.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Test Samples:</span>
                  <span className="text-white">{modelMetrics.test_samples.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Training Time:</span>
                  <span className="text-white">{modelMetrics.training_time.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Last Trained:</span>
                  <span className="text-white">{new Date(modelMetrics.last_trained).toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Feature Importance */}
            <div className="bg-gray-900 rounded-lg p-4">
              <h4 className="text-white font-medium mb-3">Top Features</h4>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {Object.entries(modelMetrics.feature_importance)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 8)
                  .map(([feature, importance]) => (
                    <div key={feature} className="flex justify-between text-sm">
                      <span className="text-gray-400 truncate">{feature.replace('_', ' ')}</span>
                      <span className="text-white">{(importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Actions */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Model Actions</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button
            onClick={testPrediction}
            disabled={!isConnected || !modelInfo || modelInfo.model_status !== 'trained'}
            className="px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
          >
            <Brain className="h-4 w-4" />
            <span>Test Prediction</span>
          </button>
          
          <button
            onClick={runBenchmark}
            disabled={!isConnected || !modelInfo || modelInfo.model_status !== 'trained'}
            className="px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
          >
            <TrendingUp className="h-4 w-4" />
            <span>Benchmark</span>
          </button>
          
          <button
            onClick={generateReport}
            disabled={!isConnected || !modelInfo || modelInfo.model_status !== 'trained'}
            className="px-4 py-3 bg-orange-600 hover:bg-orange-700 disabled:bg-orange-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
          >
            <Download className="h-4 w-4" />
            <span>Generate Report</span>
          </button>
          
          <button
            onClick={() => { loadModelInfo(); loadModelMetrics(); }}
            disabled={!isConnected}
            className="px-4 py-3 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>

        {/* Benchmark Results */}
        {benchmarkResults && (
          <div className="mt-6 p-4 bg-green-900/20 border border-green-700/50 rounded-lg">
            <h4 className="text-green-300 font-medium mb-2">Performance Benchmark Results</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Predictions/sec:</span>
                <span className="text-white ml-1">{benchmarkResults.predictions_per_second.toFixed(1)}</span>
              </div>
              <div>
                <span className="text-gray-400">Avg Time:</span>
                <span className="text-white ml-1">{benchmarkResults.avg_prediction_time_ms.toFixed(2)}ms</span>
              </div>
              <div>
                <span className="text-gray-400">Total Time:</span>
                <span className="text-white ml-1">{benchmarkResults.total_time_seconds.toFixed(2)}s</span>
              </div>
              <div>
                <span className="text-gray-400">Predictions:</span>
                <span className="text-white ml-1">{benchmarkResults.total_predictions}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Training Log */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Training Log</h3>
          <button
            onClick={() => setTrainingLog([])}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            Clear Log
          </button>
        </div>
        
        <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
          {trainingLog.length === 0 ? (
            <p className="text-gray-500">No log entries yet...</p>
          ) : (
            trainingLog.map((entry, index) => (
              <div key={index} className="text-gray-300 mb-1">
                {entry}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Model Report */}
      {modelReport && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Model Report</h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
            <pre className="text-gray-300 text-sm whitespace-pre-wrap">{modelReport}</pre>
          </div>
        </div>
      )}

      {/* Quick Start Guide */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Quick Start Guide</h3>
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="text-white font-medium mb-2">1. Start ML API Server</h4>
            <code className="text-green-400 text-sm">python ml_models/model_api.py</code>
            <p className="text-gray-400 text-xs mt-1">Starts the ML API server on port 8003</p>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="text-white font-medium mb-2">2. Install Dependencies</h4>
            <code className="text-green-400 text-sm">pip install -r ml_models/requirements.txt</code>
            <p className="text-gray-400 text-xs mt-1">Installs scikit-learn, pandas, numpy, and other ML dependencies</p>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="text-white font-medium mb-2">3. Train Model</h4>
            <p className="text-gray-300 text-sm">Use the training configuration above and click "Start Training"</p>
            <p className="text-gray-400 text-xs mt-1">Training typically takes 30-120 seconds depending on data size</p>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="text-white font-medium mb-2">4. Test & Deploy</h4>
            <p className="text-gray-300 text-sm">Run test predictions and benchmarks to validate performance</p>
            <p className="text-gray-400 text-xs mt-1">Model will automatically be used for real-time threat detection</p>
          </div>
        </div>
      </div>
    </div>
  );
}