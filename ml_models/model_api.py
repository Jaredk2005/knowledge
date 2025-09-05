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

# Import our models
from threat_detection_model import ThreatDetectionModel, threat_model
from model_trainer import ModelTrainer

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
    
    # Load existing model if available
    if not threat_model.primary_classifier:
        logging.info("No trained model found, training new model...")
        try:
            await model_trainer.train_comprehensive_model()
            logging.info("Model training completed during startup")
        except Exception as e:
            logging.error(f"Failed to train model during startup: {e}")
    
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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