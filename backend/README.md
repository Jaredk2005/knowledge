# Threat Detection Backend System

A comprehensive backend system for cybersecurity threat detection, classification, and visualization.

## Components

### 1. GeoIP Service (`geoip_service.py`)
- **Purpose**: Resolve IP addresses to geographic locations
- **Features**:
  - MaxMind GeoIP database integration
  - Online fallback services (ip-api.com)
  - Threat intelligence integration (VirusTotal, AbuseIPDB)
  - Caching for performance
  - Bulk lookup support

### 2. ML Threat Classifier (`ml_threat_classifier.py`)
- **Purpose**: Machine learning-based threat classification
- **Features**:
  - Random Forest classifier for threat categorization
  - Isolation Forest for anomaly detection
  - Feature extraction from network events
  - Rule-based fallback classification
  - Model training and persistence

### 3. Threat Ingestion API (`threat_ingestion_api.py`)
- **Purpose**: High-performance API for processing threat events
- **Features**:
  - FastAPI-based REST API
  - WebSocket support for real-time updates
  - Bulk ingestion capabilities
  - Database integration with PostgreSQL
  - Automatic geolocation and classification

### 4. Test Suite (`test_threat_system.py`)
- **Purpose**: Comprehensive testing of all components
- **Features**:
  - API endpoint testing
  - Performance benchmarking
  - Concurrent request testing
  - Detailed reporting

## Database Schema

The system uses PostgreSQL with the following main tables:

- `global_threats`: Store threat events with geolocation
- `threat_classifications`: ML classification results
- `geoip_cache`: Cached geolocation data
- `threat_statistics`: Aggregated statistics

## Setup Instructions

### Prerequisites

```bash
# Python dependencies
pip install fastapi uvicorn asyncpg geoip2 scikit-learn pandas numpy requests websockets python-multipart aiohttp

# Optional: MaxMind GeoIP database
# Download GeoLite2-City.mmdb from https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
```

### Environment Variables

Create a `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=password

# Threat Intelligence APIs (optional)
VIRUSTOTAL_API_KEY=your_virustotal_key
ABUSEIPDB_API_KEY=your_abuseipdb_key

# Supabase (for fallback)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
```

### Running the System

1. **Start the Ingestion API**:
   ```bash
   cd backend
   uvicorn threat_ingestion_api:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Test the System**:
   ```bash
   python test_threat_system.py
   ```

3. **Access the API**:
   - API Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/api/health
   - WebSocket: ws://localhost:8001/ws/threats

## API Endpoints

### Threat Ingestion
- `POST /api/threats/ingest` - Ingest single threat
- `POST /api/threats/bulk-ingest` - Bulk ingest threats

### Data Retrieval
- `GET /api/threats/global` - Get global threats for map
- `GET /api/threats/statistics` - Get threat statistics

### Real-time Updates
- `WebSocket /ws/threats` - Real-time threat notifications

## Usage Examples

### Ingest a Threat
```python
import requests

threat = {
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.5",
    "threat_type": "malware",
    "severity": 8,
    "confidence": 0.9,
    "description": "Trojan detected in network traffic"
}

response = requests.post("http://localhost:8001/api/threats/ingest", json=threat)
print(response.json())
```

### Get Global Threats
```python
import requests

response = requests.get("http://localhost:8001/api/threats/global?hours=24&min_severity=5")
threats = response.json()
print(f"Found {len(threats['threats'])} threats")
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8001/ws/threats');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'threat_detected') {
        console.log('New threat:', data.data);
    }
};
```

## Performance

The system is designed for high performance:

- **Ingestion Rate**: 100+ threats/second
- **Response Time**: <100ms for single threat ingestion
- **Concurrent Requests**: Supports 50+ concurrent connections
- **Database**: Optimized with indexes and connection pooling

## Monitoring

Monitor the system using:

- Health check endpoint: `/api/health`
- Database connection status
- WebSocket connection count
- API response times

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Check DATABASE_URL in .env
   - Ensure PostgreSQL is running
   - Verify database exists and user has permissions

2. **GeoIP Lookups Failing**:
   - Download MaxMind GeoLite2-City.mmdb
   - Check internet connection for online fallback
   - Verify API keys for threat intelligence

3. **ML Classification Errors**:
   - Models will auto-initialize on first use
   - Check disk space for model storage
   - Verify scikit-learn installation

4. **WebSocket Connection Issues**:
   - Check firewall settings
   - Verify port 8001 is accessible
   - Check browser console for errors

## Development

### Adding New Threat Types

1. Update `threat_categories` in `ml_threat_classifier.py`
2. Add classification rules in `rule_based_classification()`
3. Update frontend threat type mappings

### Extending ML Features

1. Add new features to `ThreatFeatures` dataclass
2. Update `extract_features()` method
3. Retrain models with new feature set

### Custom Threat Intelligence

1. Add new API integration to `geoip_service.py`
2. Update `_get_threat_intelligence()` method
3. Configure API keys in environment variables