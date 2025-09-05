#!/usr/bin/env python3
"""
Model Training and Management System
Comprehensive training pipeline for threat detection models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Import our threat detection model
from threat_detection_model import ThreatDetectionModel, ThreatFeatures

# Database integration
from dotenv import load_dotenv
load_dotenv()

try:
    from supabase import create_client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        SUPABASE_AVAILABLE = True
    else:
        SUPABASE_AVAILABLE = False
        print("⚠️  Supabase not configured - using synthetic data only")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  Supabase not available - using synthetic data only")

class ModelTrainer:
    """Comprehensive model training and management system"""
    
    def __init__(self, model_dir: str = "ml_models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = ThreatDetectionModel(str(self.model_dir))
        self.training_data_cache = []
        self.validation_data_cache = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.model_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def collect_training_data_from_database(self, days: int = 30) -> Tuple[List[Dict], List[str]]:
        """Collect training data from Supabase database"""
        if not SUPABASE_AVAILABLE:
            self.logger.warning("Supabase not available, using synthetic data")
            return [], []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch network traffic data
            network_response = supabase.table('network_traffic').select('*').gte(
                'timestamp', start_date.isoformat()
            ).lte('timestamp', end_date.isoformat()).execute()
            
            # Fetch threat intelligence data
            threat_response = supabase.table('threat_intelligence').select('*').eq(
                'is_active', True
            ).execute()
            
            # Fetch incidents for labeling
            incidents_response = supabase.table('incidents').select('*').gte(
                'detected_at', start_date.isoformat()
            ).execute()
            
            training_events = []
            labels = []
            
            # Process network traffic data
            if network_response.data:
                for traffic in network_response.data:
                    event_data = {
                        'source_ip': traffic['source_ip'],
                        'destination_ip': traffic['destination_ip'],
                        'protocol': traffic['protocol'],
                        'packet_size': traffic.get('packet_size', 0),
                        'bytes_transferred': traffic.get('bytes_transferred', 0),
                        'timestamp': traffic['timestamp'],
                        'is_suspicious': traffic.get('is_suspicious', False),
                        'threat_indicators': traffic.get('threat_indicators', [])
                    }
                    
                    # Label based on suspicious flag and threat indicators
                    if traffic.get('is_suspicious', False):
                        if traffic.get('threat_indicators'):
                            # Try to determine specific threat type from indicators
                            indicators = traffic.get('threat_indicators', [])
                            if any('malware' in ind.lower() for ind in indicators):
                                label = 'malware'
                            elif any('scan' in ind.lower() for ind in indicators):
                                label = 'port_scan'
                            elif any('ddos' in ind.lower() for ind in indicators):
                                label = 'ddos'
                            else:
                                label = 'malware'  # Generic threat
                        else:
                            label = 'malware'  # Generic suspicious activity
                    else:
                        label = 'benign'
                    
                    training_events.append(event_data)
                    labels.append(label)
            
            # Process incidents for additional labeled data
            if incidents_response.data:
                for incident in incidents_response.data:
                    event_data = {
                        'source_ip': incident.get('source_ip', '0.0.0.0'),
                        'destination_ip': incident.get('destination_ip', '0.0.0.0'),
                        'incident_type': incident['incident_type'],
                        'severity': incident['severity'],
                        'timestamp': incident['detected_at'],
                        'description': incident['description']
                    }
                    
                    training_events.append(event_data)
                    labels.append(incident['incident_type'])
            
            self.logger.info(f"Collected {len(training_events)} training samples from database")
            return training_events, labels
            
        except Exception as e:
            self.logger.error(f"Error collecting training data from database: {e}")
            return [], []
    
    def augment_training_data(self, events: List[Dict], labels: List[str]) -> Tuple[List[Dict], List[str]]:
        """Augment training data with variations"""
        augmented_events = events.copy()
        augmented_labels = labels.copy()
        
        # Add noise to existing samples
        for event, label in zip(events, labels):
            if label != 'benign':  # Augment threat samples more
                # Create variations
                for _ in range(2):  # Create 2 variations per threat sample
                    augmented_event = event.copy()
                    
                    # Add noise to numerical features
                    if 'packet_size' in augmented_event:
                        noise = np.random.normal(0, augmented_event['packet_size'] * 0.1)
                        augmented_event['packet_size'] = max(0, augmented_event['packet_size'] + noise)
                    
                    if 'bytes_transferred' in augmented_event:
                        noise = np.random.normal(0, augmented_event['bytes_transferred'] * 0.1)
                        augmented_event['bytes_transferred'] = max(0, augmented_event['bytes_transferred'] + noise)
                    
                    # Vary IP addresses slightly (for different subnets)
                    if 'source_ip' in augmented_event:
                        ip_parts = augmented_event['source_ip'].split('.')
                        if len(ip_parts) == 4:
                            try:
                                # Vary last octet
                                last_octet = int(ip_parts[3])
                                new_octet = max(1, min(254, last_octet + np.random.randint(-10, 11)))
                                ip_parts[3] = str(new_octet)
                                augmented_event['source_ip'] = '.'.join(ip_parts)
                            except ValueError:
                                pass
                    
                    augmented_events.append(augmented_event)
                    augmented_labels.append(label)
        
        self.logger.info(f"Augmented training data: {len(events)} -> {len(augmented_events)} samples")
        return augmented_events, augmented_labels
    
    async def train_comprehensive_model(self, use_database: bool = True, augment_data: bool = True):
        """Train comprehensive threat detection model"""
        self.logger.info("Starting comprehensive model training...")
        
        training_events = []
        labels = []
        
        # Collect data from database if available
        if use_database and SUPABASE_AVAILABLE:
            db_events, db_labels = await self.collect_training_data_from_database()
            training_events.extend(db_events)
            labels.extend(db_labels)
        
        # If we don't have enough real data, supplement with synthetic data
        if len(training_events) < 1000:
            self.logger.info("Insufficient real data, generating synthetic training data...")
            synthetic_X, synthetic_y = self.model.generate_synthetic_training_data(5000)
            
            # Convert synthetic data back to event format for consistency
            for i in range(len(synthetic_X)):
                synthetic_event = {
                    'packet_size': synthetic_X[i][0],
                    'connection_duration': synthetic_X[i][1],
                    'bytes_transferred': synthetic_X[i][2],
                    'packets_per_second': synthetic_X[i][3],
                    'unique_ports': int(synthetic_X[i][4]),
                    'source_ip': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'timestamp': datetime.now().isoformat()
                }
                training_events.append(synthetic_event)
                labels.append(synthetic_y[i])
        
        # Augment data if requested
        if augment_data:
            training_events, labels = self.augment_training_data(training_events, labels)
        
        # Train the model
        metrics = self.model.train_model(training_events, labels)
        
        # Save training data for future reference
        self.save_training_data(training_events, labels)
        
        return metrics
    
    def save_training_data(self, events: List[Dict], labels: List[str]):
        """Save training data for future use"""
        training_data = {
            'events': events,
            'labels': labels,
            'saved_at': datetime.now().isoformat(),
            'sample_count': len(events)
        }
        
        with open(self.model_dir / 'training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        self.logger.info(f"Training data saved: {len(events)} samples")
    
    def load_training_data(self) -> Tuple[List[Dict], List[str]]:
        """Load previously saved training data"""
        try:
            with open(self.model_dir / 'training_data.json', 'r') as f:
                data = json.load(f)
                return data['events'], data['labels']
        except FileNotFoundError:
            return [], []
    
    def cross_validate_model(self, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on the model"""
        if not self.primary_classifier:
            raise ValueError("Model not trained yet")
        
        # Load or generate training data
        events, labels = self.load_training_data()
        if not events:
            self.logger.info("No saved training data, generating synthetic data for CV...")
            X, y = self.model.generate_synthetic_training_data(2000)
        else:
            # Extract features
            feature_vectors = []
            for event in events:
                features = self.model.extract_features(event)
                feature_vectors.append(features.to_vector())
            X = np.array(feature_vectors)
            y = self.model.label_encoder.transform(labels)
        
        # Apply preprocessing
        if self.model.feature_selector:
            X = self.model.feature_selector.transform(X)
        X_scaled = self.model.scaler.transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model.primary_classifier, X_scaled, y, 
            cv=cv_folds, scoring='f1_weighted', n_jobs=-1
        )
        
        return {
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'scores': cv_scores.tolist(),
            'cv_folds': cv_folds
        }
    
    def benchmark_model_performance(self, num_predictions: int = 1000) -> Dict[str, float]:
        """Benchmark model prediction performance"""
        if not self.model.primary_classifier:
            raise ValueError("Model not trained yet")
        
        # Generate test events
        test_events = []
        for _ in range(num_predictions):
            event = {
                'source_ip': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'destination_ip': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'packet_size': np.random.randint(64, 1500),
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
                'timestamp': datetime.now().isoformat()
            }
            test_events.append(event)
        
        # Benchmark prediction time
        start_time = datetime.now()
        
        for event in test_events:
            self.model.predict_threat(event)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return {
            'total_predictions': num_predictions,
            'total_time_seconds': total_time,
            'predictions_per_second': num_predictions / total_time,
            'avg_prediction_time_ms': (total_time / num_predictions) * 1000
        }
    
    def generate_model_report(self) -> str:
        """Generate comprehensive model report"""
        report_lines = [
            "THREAT DETECTION MODEL REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Model information
        model_info = self.model.get_model_info()
        report_lines.extend([
            "MODEL INFORMATION:",
            f"Status: {model_info['model_status']}",
            f"Features: {model_info['feature_count']}",
            f"Threat Categories: {len(model_info['threat_categories'])}",
            ""
        ])
        
        # Performance metrics
        if self.model.metrics:
            metrics = self.model.metrics
            report_lines.extend([
                "PERFORMANCE METRICS:",
                f"Accuracy: {metrics.accuracy:.3f}",
                f"Precision: {metrics.precision:.3f}",
                f"Recall: {metrics.recall:.3f}",
                f"F1 Score: {metrics.f1_score:.3f}",
                f"AUC Score: {metrics.auc_score:.3f}",
                f"Training Samples: {metrics.training_samples:,}",
                f"Test Samples: {metrics.test_samples:,}",
                f"Training Time: {metrics.training_time:.2f}s",
                ""
            ])
            
            # Feature importance
            report_lines.append("TOP 10 IMPORTANT FEATURES:")
            sorted_features = sorted(
                metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report_lines.append(f"{i:2d}. {feature}: {importance:.3f}")
            report_lines.append("")
        
        # Cross-validation results
        try:
            cv_results = self.cross_validate_model()
            report_lines.extend([
                "CROSS-VALIDATION RESULTS:",
                f"Mean F1 Score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}",
                f"CV Folds: {cv_results['cv_folds']}",
                ""
            ])
        except Exception as e:
            report_lines.append(f"Cross-validation failed: {e}")
            report_lines.append("")
        
        # Performance benchmark
        try:
            benchmark = self.benchmark_model_performance(100)
            report_lines.extend([
                "PERFORMANCE BENCHMARK:",
                f"Predictions per second: {benchmark['predictions_per_second']:.1f}",
                f"Average prediction time: {benchmark['avg_prediction_time_ms']:.2f}ms",
                ""
            ])
        except Exception as e:
            report_lines.append(f"Performance benchmark failed: {e}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_model_report(self):
        """Save model report to file"""
        report = self.generate_model_report()
        report_path = self.model_dir / f'model_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Model report saved to {report_path}")
        return report_path
    
    async def continuous_learning_pipeline(self, check_interval: int = 3600):
        """Continuous learning pipeline that retrains model with new data"""
        self.logger.info("Starting continuous learning pipeline...")
        
        while True:
            try:
                # Check for new data
                new_events, new_labels = await self.collect_training_data_from_database(days=1)
                
                if len(new_events) > 10:  # Only retrain if we have sufficient new data
                    self.logger.info(f"Retraining model with {len(new_events)} new samples...")
                    
                    # Retrain model
                    metrics = self.model.retrain_with_new_data(new_events, new_labels)
                    
                    # Log performance
                    self.logger.info(f"Model retrained - Accuracy: {metrics.accuracy:.3f}")
                    
                    # Save updated report
                    self.save_model_report()
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning pipeline: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def export_model_for_production(self, export_path: str):
        """Export model for production deployment"""
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        import shutil
        
        model_files = [
            'random_forest_classifier.pkl',
            'isolation_forest.pkl',
            'feature_selector.pkl',
            'feature_scaler.pkl',
            'label_encoder.pkl',
            'model_metadata.json'
        ]
        
        for file_name in model_files:
            src_path = self.model_dir / file_name
            if src_path.exists():
                shutil.copy2(src_path, export_dir / file_name)
        
        # Create production configuration
        prod_config = {
            'model_version': '2.0',
            'exported_at': datetime.now().isoformat(),
            'feature_names': self.model.feature_names,
            'threat_categories': self.model.threat_categories,
            'performance_metrics': asdict(self.model.metrics) if self.model.metrics else None
        }
        
        with open(export_dir / 'production_config.json', 'w') as f:
            json.dump(prod_config, f, indent=2)
        
        self.logger.info(f"Model exported for production to {export_dir}")

async def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threat Detection Model Trainer')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--retrain', action='store_true', help='Retrain with new data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--report', action='store_true', help='Generate model report')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model performance')
    parser.add_argument('--export', type=str, help='Export model for production')
    parser.add_argument('--continuous', action='store_true', help='Start continuous learning')
    parser.add_argument('--days', type=int, default=30, help='Days of data to collect')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.train:
        print("🤖 Training new threat detection model...")
        metrics = await trainer.train_comprehensive_model()
        print(f"✅ Training completed!")
        print(f"   Accuracy: {metrics.accuracy:.3f}")
        print(f"   F1 Score: {metrics.f1_score:.3f}")
        print(f"   Training time: {metrics.training_time:.2f}s")
        
        # Generate report
        trainer.save_model_report()
    
    elif args.retrain:
        print("🔄 Retraining model with new data...")
        new_events, new_labels = await trainer.collect_training_data_from_database(args.days)
        if new_events:
            metrics = trainer.model.retrain_with_new_data(new_events, new_labels)
            print(f"✅ Retraining completed with {len(new_events)} new samples!")
        else:
            print("❌ No new training data found")
    
    elif args.evaluate:
        print("📊 Evaluating model...")
        try:
            cv_results = trainer.cross_validate_model()
            print(f"Cross-validation F1 Score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    elif args.report:
        print("📄 Generating model report...")
        report_path = trainer.save_model_report()
        print(f"✅ Report saved to {report_path}")
    
    elif args.benchmark:
        print("⚡ Benchmarking model performance...")
        try:
            benchmark = trainer.benchmark_model_performance()
            print(f"Predictions per second: {benchmark['predictions_per_second']:.1f}")
            print(f"Average prediction time: {benchmark['avg_prediction_time_ms']:.2f}ms")
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    elif args.export:
        print(f"📦 Exporting model to {args.export}...")
        trainer.export_model_for_production(args.export)
        print("✅ Model exported successfully!")
    
    elif args.continuous:
        print("🔄 Starting continuous learning pipeline...")
        print("   Press Ctrl+C to stop")
        try:
            await trainer.continuous_learning_pipeline()
        except KeyboardInterrupt:
            print("\n🛑 Continuous learning stopped")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())