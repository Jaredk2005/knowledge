# Machine Learning Threat Detection Models

Advanced machine learning system for cybersecurity threat detection using Random Forest classification and anomaly detection.

## Overview

This ML system provides:
- **Random Forest Classification**: Multi-class threat categorization
- **Isolation Forest**: Anomaly detection for unknown threats
- **Feature Engineering**: 25+ advanced features for threat analysis
- **Continuous Learning**: Automated retraining with new data
- **Real-time Inference**: High-performance prediction API
- **Model Management**: Training, evaluation, and deployment tools

## Components

### 1. Core Model (`threat_detection_model.py`)
- **ThreatDetectionModel**: Main ML model class
- **ThreatFeatures**: Comprehensive feature extraction
- **Random Forest Classifier**: Primary threat classification
- **Isolation Forest**: Anomaly detection
- **Feature Engineering**: Advanced feature extraction from network events

### 2. Training Pipeline (`model_trainer.py`)
- **ModelTrainer**: Training and management system
- **Data Collection**: Automated data gathering from Supabase
- **Data Augmentation**: Synthetic data generation and augmentation
- **Hyperparameter Tuning**: Automated parameter optimization
- **Continuous Learning**: Automated retraining pipeline

### 3. API Server (`model_api.py`)
- **FastAPI Server**: REST API for model inference
- **Batch Processing**: Efficient batch threat prediction
- **Model Management**: Training and evaluation endpoints
- **Performance Monitoring**: Benchmarking and metrics

## Features Extracted

The model extracts 25+ features from network events:

### Network Features
- Packet size and connection duration
- Bytes transferred and packet rate
- Unique ports and protocol diversity

### Behavioral Features
- Connection frequency patterns
- Geographic distance analysis
- IP reputation scoring

### Content Features
- Payload entropy analysis
- Suspicious string detection
- Base64 and encryption detection

### Temporal Features
- Time-based patterns
- Day/hour analysis
- Weekend activity detection

### Advanced Features
- TCP flags anomaly detection
- Port scan indicators
- DDoS pattern recognition
- Malware signature counting

## Threat Categories

The model classifies threats into 13 categories:
- `benign` - Normal traffic
- `malware` - Malicious software
- `botnet` - Botnet activity
- `ddos` - Distributed denial of service
- `brute_force` - Password attacks
- `sql_injection` - Database attacks
- `xss` - Cross-site scripting
- `phishing` - Social engineering
- `ransomware` - Encryption attacks
- `apt` - Advanced persistent threats
- `port_scan` - Network reconnaissance
- `data_exfiltration` - Data theft
- `lateral_movement` - Network traversal

## Installation

```bash
# Install Python dependencies
pip install scikit-learn pandas numpy fastapi uvicorn joblib matplotlib seaborn

# Install optional dependencies for enhanced features
pip install supabase python-dotenv asyncio

# Create model directory
mkdir -p ml_models/saved_models
```

## Usage

### 1. Train the Model

```bash
# Train new model with synthetic data
python ml_models/model_trainer.py --train

# Train with database data (requires Supabase)
python ml_models/model_trainer.py --train --days 30

# Generate model report
python ml_models/model_trainer.py --report
```

### 2. Start the API Server

```bash
# Start ML API server on port 8003
python ml_models/model_api.py
```

### 3. Make Predictions

```python
import requests

# Single prediction
threat_data = {
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.5",
    "packet_size": 1024,
    "protocol": "TCP",
    "payload": "suspicious content here"
}

response = requests.post("http://localhost:8003/api/ml/predict", json=threat_data)
result = response.json()

print(f"Threat Type: {result['threat_type']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Severity: {result['severity']}")
```

### 4. Batch Processing

```python
# Batch prediction for multiple events
events = [
    {"source_ip": "192.168.1.100", "packet_size": 1024},
    {"source_ip": "10.0.0.50", "packet_size": 2048},
    # ... more events
]

response = requests.post("http://localhost:8003/api/ml/predict/batch", json=events)
results = response.json()
```

## API Endpoints

### Prediction Endpoints
- `POST /api/ml/predict` - Single threat prediction
- `POST /api/ml/predict/batch` - Batch threat prediction

### Model Management
- `GET /api/ml/model/info` - Model information
- `POST /api/ml/model/train` - Train/retrain model
- `GET /api/ml/model/metrics` - Performance metrics
- `GET /api/ml/model/report` - Comprehensive report
- `POST /api/ml/model/benchmark` - Performance benchmark

### System
- `GET /api/ml/health` - Health check

## Model Performance

### Expected Performance Metrics
- **Accuracy**: 85-95% on balanced datasets
- **Precision**: 80-90% for threat detection
- **Recall**: 85-95% for critical threats
- **F1 Score**: 85-92% weighted average
- **Inference Speed**: 100+ predictions/second

### Feature Importance
Top contributing features typically include:
1. Payload entropy
2. Suspicious string count
3. Reputation score
4. Packet rate
5. Connection frequency

## Training Data

### Synthetic Data Generation
The system can generate realistic synthetic training data:
- **Benign Traffic**: Normal network patterns
- **Malware**: Suspicious payload patterns
- **DDoS**: High packet rate, small payloads
- **Port Scans**: Sequential port access
- **Brute Force**: Repeated connection attempts

### Real Data Integration
- Collects data from Supabase tables
- Processes network traffic logs
- Incorporates incident reports
- Uses threat intelligence feeds

## Continuous Learning

### Automated Retraining
```bash
# Start continuous learning pipeline
python ml_models/model_trainer.py --continuous
```

Features:
- Monitors for new threat data
- Automatically retrains when sufficient new data available
- Validates model performance after retraining
- Generates updated performance reports

### Model Versioning
- Automatic model versioning
- Performance comparison between versions
- Rollback capabilities
- A/B testing support

## Integration

### Backend Integration
```python
from ml_models.threat_detection_model import threat_model

# Use in existing threat detection systems
event_data = {
    'source_ip': '192.168.1.100',
    'packet_size': 1024,
    'protocol': 'TCP'
}

result = threat_model.predict_threat(event_data)
if result['final_classification']['is_threat']:
    print(f"Threat detected: {result['final_classification']['threat_type']}")
```

### Frontend Integration
```javascript
// Call ML API from frontend
const predictThreat = async (eventData) => {
  const response = await fetch('http://localhost:8003/api/ml/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(eventData)
  });
  
  return await response.json();
};
```

## Monitoring and Evaluation

### Model Monitoring
- Real-time performance tracking
- Prediction confidence monitoring
- Feature drift detection
- Model degradation alerts

### Evaluation Metrics
- Cross-validation scores
- Confusion matrix analysis
- ROC/AUC curves
- Precision-recall curves

## Production Deployment

### Export for Production
```bash
# Export trained model for production
python ml_models/model_trainer.py --export /path/to/production/models
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

COPY ml_models/ /app/ml_models/
COPY requirements.txt /app/

WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8003
CMD ["python", "ml_models/model_api.py"]
```

## Troubleshooting

### Common Issues

1. **Model Not Training**
   - Check if dependencies are installed
   - Verify training data availability
   - Check disk space for model storage

2. **Poor Performance**
   - Increase training data size
   - Adjust hyperparameters
   - Check feature quality

3. **Slow Predictions**
   - Reduce feature count
   - Optimize preprocessing
   - Use model compression

### Performance Optimization
- Feature selection reduces dimensionality
- Model compression for faster inference
- Batch processing for multiple predictions
- Caching for repeated predictions

## Security Considerations

- Model files are stored securely
- API endpoints require authentication in production
- Training data is sanitized and validated
- Model predictions are logged for audit

## Future Enhancements

- Deep learning models (LSTM, CNN)
- Federated learning capabilities
- Real-time model updates
- Advanced ensemble methods
- Explainable AI features