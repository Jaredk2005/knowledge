#!/usr/bin/env python3
"""
ML Model API Server
REST API for threat detection model inference and management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
import pandas as pd
import io

# Import our models
from threat_detection_model import ThreatDetectionModel, threat_model
from model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ThreatPredictionRequest(BaseModel):
    source_ip: str
    destination_ip: Optional[str] = None
    packet_size: Optional[int] = 0
    protocol: Optional[str] = "TCP"
    payload: Optional[str] = ""
    timestamp: Optional[str] = None
    additional_features: Optional[Dict[str, Any]] = {}

class ThreatPredictionResponse(BaseModel):
    threat_type: str
    confidence: float
    severity: int
    is_threat: bool
    classification_method: str
    anomaly_detected: bool
    rules_triggered: List[str]
    processing_time_ms: float

class ModelTrainingRequest(BaseModel):
    use_database: bool = True
    augment_data: bool = True
    training_samples: Optional[int] = None

class CSVTrainingRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    label_column: str
    feature_columns: List[str]
    use_csv: bool = True
class ModelInfo(BaseModel):
    model_status: str
    feature_count: int
    threat_categories: List[str]
    last_trained: Optional[str] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None

# Global trainer instance
model_trainer = ModelTrainer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logging.info("ML Model API starting up...")
    
    # Initialize model without auto-training
    logging.info("ML Model API ready for training requests")
    
    yield
    
    # Shutdown
    logging.info("ML Model API shutting down...")

# FastAPI app
app = FastAPI(
    title="Threat Detection ML API",
    description="Machine Learning API for cybersecurity threat detection and classification",
    version="2.0.0",
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

@app.post("/api/ml/predict", response_model=ThreatPredictionResponse)
async def predict_threat(request: ThreatPredictionRequest):
    """Predict threat classification for network event"""
    try:
        start_time = datetime.now()
        
        # Prepare event data
        event_data = {
            'source_ip': request.source_ip,
            'destination_ip': request.destination_ip,
            'packet_size': request.packet_size,
            'protocol': request.protocol,
            'payload': request.payload.encode() if request.payload else b'',
            'timestamp': request.timestamp or datetime.now().isoformat(),
            **request.additional_features
        }
        
        # Make prediction
        result = threat_model.predict_threat(event_data)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract results
        final_classification = result.get('final_classification', {})
        anomaly_detection = result.get('anomaly_detection', {})
        rule_based = result.get('rule_based', {})
        
        return ThreatPredictionResponse(
            threat_type=final_classification.get('threat_type', 'unknown'),
            confidence=final_classification.get('confidence', 0.0),
            severity=final_classification.get('severity', 1),
            is_threat=final_classification.get('is_threat', False),
            classification_method=final_classification.get('classification_method', 'unknown'),
            anomaly_detected=anomaly_detection.get('is_anomaly', False),
            rules_triggered=rule_based.get('rules_triggered', []),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/ml/predict/batch")
async def predict_threats_batch(requests: List[ThreatPredictionRequest]):
    """Batch threat prediction for multiple events"""
    try:
        results = []
        
        for req in requests:
            event_data = {
                'source_ip': req.source_ip,
                'destination_ip': req.destination_ip,
                'packet_size': req.packet_size,
                'protocol': req.protocol,
                'payload': req.payload.encode() if req.payload else b'',
                'timestamp': req.timestamp or datetime.now().isoformat(),
                **req.additional_features
            }
            
            result = threat_model.predict_threat(event_data)
            final_classification = result.get('final_classification', {})
            
            results.append({
                'source_ip': req.source_ip,
                'threat_type': final_classification.get('threat_type', 'unknown'),
                'confidence': final_classification.get('confidence', 0.0),
                'severity': final_classification.get('severity', 1),
                'is_threat': final_classification.get('is_threat', False)
            })
        
        return {
            'predictions': results,
            'total_processed': len(results)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/ml/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    try:
        info = threat_model.get_model_info()
        
        return ModelInfo(
            model_status=info['model_status'],
            feature_count=info['feature_count'],
            threat_categories=info['threat_categories'],
            last_trained=info.get('performance', {}).get('last_trained'),
            accuracy=info.get('performance', {}).get('accuracy'),
            f1_score=info.get('performance', {}).get('f1_score')
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/model/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain the threat detection model"""
    try:
        # Start training in background
        background_tasks.add_task(
            model_trainer.train_comprehensive_model,
            request.use_database,
            request.augment_data
        )
        
        return {
            'status': 'training_started',
            'message': 'Model training started in background',
            'use_database': request.use_database,
            'augment_data': request.augment_data
        }
        
    except Exception as e:
        logger.error(f"Training initiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.post("/api/ml/model/train-csv")
async def train_model_with_csv(request: CSVTrainingRequest, background_tasks: BackgroundTasks):
    """Train model with uploaded CSV data"""
    try:
        # Validate CSV data
        if not request.training_data:
            raise HTTPException(status_code=400, detail="No training data provided")
        
        if not request.label_column:
            raise HTTPException(status_code=400, detail="Label column not specified")
        
        if not request.feature_columns:
            raise HTTPException(status_code=400, detail="No feature columns specified")
        
        # Validate that columns exist in data
        sample_row = request.training_data[0]
        if request.label_column not in sample_row:
            raise HTTPException(status_code=400, detail=f"Label column '{request.label_column}' not found in data")
        
        missing_features = [col for col in request.feature_columns if col not in sample_row]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing_features}")
        
        # Process CSV data for training
        def train_with_csv_data():
            try:
                # Convert to training format
                training_events = []
                labels = []
                
                for row in request.training_data:
                    # Extract label
                    label = row[request.label_column]
                    labels.append(label)
                    
                    # Create event data from features
                    event_data = {
                        'source_ip': row.get('source_ip', '192.168.1.1'),
                        'destination_ip': row.get('destination_ip', '10.0.0.1'),
                        'packet_size': float(row.get('packet_size', 0)),
                        'protocol': row.get('protocol', 'TCP'),
                        'bytes_transferred': float(row.get('bytes_transferred', 0)),
                        'timestamp': row.get('timestamp', datetime.now().isoformat()),
                        'payload': row.get('payload', '').encode() if row.get('payload') else b'',
                        'source_port': int(row.get('source_port', 0)) if row.get('source_port') else 0,
                        'destination_port': int(row.get('destination_port', 0)) if row.get('destination_port') else 0,
                        'connection_duration': float(row.get('connection_duration', 0)),
                        'packets_per_second': float(row.get('packets_per_second', 1)),
                        'reputation_score': float(row.get('reputation_score', 0.5))
                    }
                    
                    # Add any additional features from CSV
                    for col in request.feature_columns:
                        if col not in event_data and col in row:
                            try:
                                # Try to convert to float, fallback to string
                                event_data[col] = float(row[col])
                            except (ValueError, TypeError):
                                event_data[col] = str(row[col])
                    
                    training_events.append(event_data)
                
                # Train the model
                logger.info(f"Training model with {len(training_events)} CSV samples")
                metrics = threat_model.train_model(training_events, labels)
                logger.info(f"CSV training completed - Accuracy: {metrics.accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"CSV training error: {e}")
        
        # Start training in background
        background_tasks.add_task(train_with_csv_data)
        
        return {
            'status': 'csv_training_started',
            'message': f'Model training started with {len(request.training_data)} CSV samples',
            'label_column': request.label_column,
            'feature_columns': request.feature_columns,
            'sample_count': len(request.training_data)
        }
        
    except Exception as e:
        logger.error(f"CSV training initiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start CSV training: {str(e)}")
@app.get("/api/ml/model/metrics")
async def get_model_metrics():
    """Get detailed model performance metrics"""
    try:
        if not threat_model.metrics:
            raise HTTPException(status_code=404, detail="No model metrics available - model not trained")
        
        return {
            'accuracy': threat_model.metrics.accuracy,
            'precision': threat_model.metrics.precision,
            'recall': threat_model.metrics.recall,
            'f1_score': threat_model.metrics.f1_score,
            'auc_score': threat_model.metrics.auc_score,
            'training_samples': threat_model.metrics.training_samples,
            'test_samples': threat_model.metrics.test_samples,
            'training_time': threat_model.metrics.training_time,
            'last_trained': threat_model.metrics.last_trained,
            'feature_importance': threat_model.metrics.feature_importance,
            'confusion_matrix': threat_model.metrics.confusion_matrix
        }
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/model/report")
async def get_model_report():
    """Get comprehensive model report"""
    try:
        report = model_trainer.generate_model_report()
        return {'report': report}
    except Exception as e:
        logger.error(f"Error generating model report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/model/benchmark")
async def benchmark_model():
    """Benchmark model performance"""
    try:
        benchmark_results = model_trainer.benchmark_model_performance()
        return benchmark_results
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': threat_model.primary_classifier is not None,
        'anomaly_detector_loaded': threat_model.anomaly_detector is not None
    }

if __name__ == "__main__":
    print("🤖 Threat Detection ML Model API")
    print("=" * 40)
    print("🧠 Advanced machine learning threat classification")
    print("🎯 Random Forest and Isolation Forest models")
    print("📊 Real-time prediction and model management")
    print("🔄 Continuous learning capabilities")
    print()
    print("API Endpoints:")
    print("  POST /api/ml/predict - Predict single threat")
    print("  POST /api/ml/predict/batch - Batch prediction")
    print("  GET  /api/ml/model/info - Model information")
    print("  POST /api/ml/model/train - Train/retrain model")
    print("  GET  /api/ml/model/metrics - Performance metrics")
    print("  GET  /api/ml/model/report - Comprehensive report")
    print("  POST /api/ml/model/benchmark - Performance benchmark")
    print("  GET  /api/ml/health - Health check")
    print()
    
    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )