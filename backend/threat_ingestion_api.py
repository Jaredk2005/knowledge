#!/usr/bin/env python3
"""
Threat Ingestion API
High-performance API for processing and storing threat events
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
import logging
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager
import asyncpg
import os
from dotenv import load_dotenv

# Import our custom modules
from geoip_service import geoip_service, GeoLocation
from ml_threat_classifier import threat_classifier

load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL") or f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'postgres')}"

# Pydantic models
class ThreatEvent(BaseModel):
    source_ip: str
    destination_ip: Optional[str] = None
    threat_type: str
    severity: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    protocol: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    packet_size: Optional[int] = None
    payload: Optional[str] = None
    timestamp: Optional[datetime] = None
    raw_data: Optional[Dict[str, Any]] = None

class ThreatResponse(BaseModel):
    id: str
    status: str
    message: str
    classification: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, Any]] = None

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logging.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Global WebSocket manager
websocket_manager = WebSocketManager()

# Database connection pool
db_pool = None

async def get_db_pool():
    """Get database connection pool"""
    global db_pool
    if db_pool is None:
        try:
            db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logging.info("Database connection pool created")
        except Exception as e:
            logging.error(f"Failed to create database pool: {e}")
            raise
    return db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await get_db_pool()
    logging.info("Threat Ingestion API started")
    
    yield
    
    # Shutdown
    if db_pool:
        await db_pool.close()
    logging.info("Threat Ingestion API stopped")

# FastAPI app
app = FastAPI(
    title="Threat Ingestion API",
    description="High-performance API for processing and storing threat events",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_threat_event(event: ThreatEvent) -> Dict[str, Any]:
    """Process a single threat event"""
    try:
        # Generate unique threat ID
        threat_id = str(uuid.uuid4())
        
        # Set timestamp if not provided
        if not event.timestamp:
            event.timestamp = datetime.utcnow()
        
        # Get geolocation data
        source_location = await geoip_service.get_location(event.source_ip)
        dest_location = None
        if event.destination_ip:
            dest_location = await geoip_service.get_location(event.destination_ip)
        
        # Prepare event data for ML classification
        ml_event_data = {
            'source_ip': event.source_ip,
            'destination_ip': event.destination_ip,
            'packet_size': event.packet_size or 0,
            'payload': event.payload.encode() if event.payload else b'',
            'protocol': event.protocol or 'TCP',
            'source_port': event.source_port or 0,
            'destination_port': event.destination_port or 0,
            'timestamp': event.timestamp,
            'reputation_score': source_location.reputation_score if source_location else 0.5,
            'country_risk_score': 0.5,  # Could be enhanced with country risk data
            'is_tor_exit': False,  # Could be enhanced with Tor detection
            'is_vpn': False,  # Could be enhanced with VPN detection
        }
        
        # Classify threat using ML
        classification_result = threat_classifier.classify_threat(ml_event_data)
        
        # Store in database
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Insert into global_threats table
            query = """
                INSERT INTO global_threats (
                    threat_id, source_ip, destination_ip, threat_type, severity, confidence,
                    description, source_country, source_country_code, source_city,
                    source_latitude, source_longitude, dest_country, dest_country_code,
                    dest_city, dest_latitude, dest_longitude, ml_classification,
                    threat_indicators, attack_vector, status, first_seen, last_seen,
                    raw_event_data
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
                ) RETURNING id
            """
            
            threat_indicators = []
            if classification_result.get('rule_based', {}).get('rules_triggered'):
                threat_indicators = classification_result['rule_based']['rules_triggered']
            
            # Determine attack vector from classification
            attack_vector = classification_result.get('final_classification', {}).get('threat_type', 'unknown')
            
            threat_db_id = await conn.fetchval(
                query,
                threat_id,
                event.source_ip,
                event.destination_ip,
                event.threat_type,
                event.severity,
                event.confidence,
                event.description,
                source_location.country if source_location else None,
                source_location.country_code if source_location else None,
                source_location.city if source_location else None,
                source_location.latitude if source_location else None,
                source_location.longitude if source_location else None,
                dest_location.country if dest_location else None,
                dest_location.country_code if dest_location else None,
                dest_location.city if dest_location else None,
                dest_location.latitude if dest_location else None,
                dest_location.longitude if dest_location else None,
                json.dumps(classification_result),
                threat_indicators,
                attack_vector,
                'active',
                event.timestamp,
                event.timestamp,
                json.dumps(event.raw_data) if event.raw_data else None
            )
            
            # Insert classification details
            if classification_result:
                classification_query = """
                    INSERT INTO threat_classifications (
                        threat_id, model_name, model_version, classification_result,
                        confidence_score, features_used
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """
                
                await conn.execute(
                    classification_query,
                    threat_db_id,
                    'ml_classifier',
                    classification_result.get('model_version', '1.0'),
                    json.dumps(classification_result),
                    classification_result.get('final_classification', {}).get('confidence', 0.5),
                    json.dumps({'feature_count': classification_result.get('features_used', 0)})
                )
        
        # Prepare response data
        response_data = {
            'id': threat_id,
            'status': 'processed',
            'message': 'Threat event processed successfully',
            'classification': classification_result,
            'location': {
                'source': source_location.__dict__ if source_location else None,
                'destination': dest_location.__dict__ if dest_location else None
            }
        }
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast({
            'type': 'threat_detected',
            'data': {
                'id': threat_id,
                'source_ip': event.source_ip,
                'destination_ip': event.destination_ip,
                'threat_type': event.threat_type,
                'severity': event.severity,
                'confidence': event.confidence,
                'description': event.description,
                'timestamp': event.timestamp.isoformat(),
                'location': response_data['location'],
                'classification': classification_result.get('final_classification', {})
            }
        })
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing threat event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process threat event: {str(e)}")

@app.post("/api/threats/ingest", response_model=ThreatResponse)
async def ingest_threat(event: ThreatEvent, background_tasks: BackgroundTasks):
    """Ingest a single threat event"""
    try:
        result = await process_threat_event(event)
        return ThreatResponse(**result)
    except Exception as e:
        logger.error(f"Threat ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threats/bulk-ingest")
async def bulk_ingest_threats(events: List[ThreatEvent]):
    """Ingest multiple threat events"""
    try:
        results = []
        for event in events:
            try:
                result = await process_threat_event(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process event from {event.source_ip}: {e}")
                results.append({
                    'id': None,
                    'status': 'failed',
                    'message': str(e),
                    'source_ip': event.source_ip
                })
        
        return {
            'processed': len([r for r in results if r['status'] == 'processed']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threats/global")
async def get_global_threats(
    limit: int = 1000,
    hours: int = 24,
    min_severity: int = 1,
    country: Optional[str] = None
):
    """Get global threats for map display"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Build query with filters
            where_conditions = ["first_seen >= $1", "severity >= $2", "status = 'active'"]
            params = [datetime.utcnow() - timedelta(hours=hours), min_severity]
            param_count = 2
            
            if country:
                param_count += 1
                where_conditions.append(f"source_country_code = ${param_count}")
                params.append(country)
            
            query = f"""
                SELECT 
                    threat_id, source_ip, destination_ip, threat_type, severity, confidence,
                    description, source_country, source_country_code, source_city,
                    source_latitude, source_longitude, dest_country, dest_country_code,
                    dest_city, dest_latitude, dest_longitude, attack_vector,
                    first_seen, last_seen, event_count
                FROM global_threats
                WHERE {' AND '.join(where_conditions)}
                ORDER BY first_seen DESC
                LIMIT ${param_count + 1}
            """
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            threats = []
            for row in rows:
                threat = {
                    'id': row['threat_id'],
                    'source_ip': str(row['source_ip']),
                    'destination_ip': str(row['destination_ip']) if row['destination_ip'] else None,
                    'threat_type': row['threat_type'],
                    'severity': row['severity'],
                    'confidence': float(row['confidence']),
                    'description': row['description'],
                    'attack_vector': row['attack_vector'],
                    'first_seen': row['first_seen'].isoformat(),
                    'last_seen': row['last_seen'].isoformat(),
                    'event_count': row['event_count'],
                    'source_location': {
                        'country': row['source_country'],
                        'country_code': row['source_country_code'],
                        'city': row['source_city'],
                        'latitude': float(row['source_latitude']) if row['source_latitude'] else None,
                        'longitude': float(row['source_longitude']) if row['source_longitude'] else None
                    },
                    'destination_location': {
                        'country': row['dest_country'],
                        'country_code': row['dest_country_code'],
                        'city': row['dest_city'],
                        'latitude': float(row['dest_latitude']) if row['dest_latitude'] else None,
                        'longitude': float(row['dest_longitude']) if row['dest_longitude'] else None
                    } if row['dest_latitude'] and row['dest_longitude'] else None
                }
                threats.append(threat)
            
            return {
                'threats': threats,
                'total': len(threats),
                'filters': {
                    'hours': hours,
                    'min_severity': min_severity,
                    'country': country
                }
            }
    except Exception as e:
        logger.error(f"Failed to get global threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threats/statistics")
async def get_threat_statistics():
    """Get threat statistics for dashboard"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Get basic statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_threats,
                    COUNT(CASE WHEN first_seen >= NOW() - INTERVAL '1 hour' THEN 1 END) as threats_last_hour,
                    COUNT(CASE WHEN first_seen >= NOW() - INTERVAL '24 hours' THEN 1 END) as threats_last_24h,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_threats,
                    AVG(severity) as avg_severity,
                    COUNT(DISTINCT source_country_code) as countries_affected
                FROM global_threats
                WHERE first_seen >= NOW() - INTERVAL '7 days'
            """
            
            stats = await conn.fetchrow(stats_query)
            
            # Get threat types distribution
            types_query = """
                SELECT threat_type, COUNT(*) as count
                FROM global_threats
                WHERE first_seen >= NOW() - INTERVAL '24 hours'
                GROUP BY threat_type
                ORDER BY count DESC
                LIMIT 10
            """
            
            threat_types = await conn.fetch(types_query)
            
            # Get top countries
            countries_query = """
                SELECT source_country, source_country_code, COUNT(*) as count
                FROM global_threats
                WHERE first_seen >= NOW() - INTERVAL '24 hours'
                  AND source_country IS NOT NULL
                GROUP BY source_country, source_country_code
                ORDER BY count DESC
                LIMIT 10
            """
            
            countries = await conn.fetch(countries_query)
            
            # Get severity distribution
            severity_query = """
                SELECT severity, COUNT(*) as count
                FROM global_threats
                WHERE first_seen >= NOW() - INTERVAL '24 hours'
                GROUP BY severity
                ORDER BY severity
            """
            
            severity_dist = await conn.fetch(severity_query)
            
            return {
                'overview': {
                    'total_threats': stats['total_threats'],
                    'threats_last_hour': stats['threats_last_hour'],
                    'threats_last_24h': stats['threats_last_24h'],
                    'active_threats': stats['active_threats'],
                    'avg_severity': float(stats['avg_severity']) if stats['avg_severity'] else 0,
                    'countries_affected': stats['countries_affected']
                },
                'threat_types': [
                    {'type': row['threat_type'], 'count': row['count']}
                    for row in threat_types
                ],
                'top_countries': [
                    {
                        'country': row['source_country'],
                        'country_code': row['source_country_code'],
                        'count': row['count']
                    }
                    for row in countries
                ],
                'severity_distribution': [
                    {'severity': row['severity'], 'count': row['count']}
                    for row in severity_dist
                ]
            }
    except Exception as e:
        logger.error(f"Failed to get threat statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/threats")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time threat updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'websocket_connections': len(websocket_manager.active_connections)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "threat_ingestion_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )