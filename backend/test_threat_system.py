#!/usr/bin/env python3
"""
Test script for the complete threat detection system
Tests all components: GeoIP, ML Classification, and Ingestion API
"""

import asyncio
import json
import random
import time
from datetime import datetime
import requests
from typing import List, Dict

# Test data
SAMPLE_THREATS = [
    {
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.5",
        "threat_type": "malware",
        "severity": 8,
        "confidence": 0.9,
        "description": "Trojan detected in network traffic",
        "protocol": "TCP",
        "source_port": 12345,
        "destination_port": 80,
        "packet_size": 1024,
        "payload": "suspicious malware payload data"
    },
    {
        "source_ip": "172.16.0.50",
        "destination_ip": "8.8.8.8",
        "threat_type": "ddos",
        "severity": 9,
        "confidence": 0.95,
        "description": "DDoS attack detected",
        "protocol": "UDP",
        "source_port": 53,
        "destination_port": 53,
        "packet_size": 1500
    },
    {
        "source_ip": "10.0.0.99",
        "destination_ip": "192.168.1.1",
        "threat_type": "brute_force",
        "severity": 7,
        "confidence": 0.85,
        "description": "Brute force login attempt",
        "protocol": "TCP",
        "source_port": 22,
        "destination_port": 22,
        "packet_size": 512
    },
    {
        "source_ip": "203.0.113.45",
        "destination_ip": "192.168.1.10",
        "threat_type": "sql_injection",
        "severity": 8,
        "confidence": 0.88,
        "description": "SQL injection attempt detected",
        "protocol": "HTTP",
        "source_port": 80,
        "destination_port": 3306,
        "packet_size": 2048,
        "payload": "'; DROP TABLE users; --"
    }
]

class ThreatSystemTester:
    def __init__(self, api_base_url: str = "http://localhost:8001/api"):
        self.api_base_url = api_base_url
        self.test_results = {
            "ingestion": [],
            "retrieval": [],
            "websocket": [],
            "performance": {}
        }
    
    async def test_threat_ingestion(self):
        """Test threat ingestion API"""
        print("🧪 Testing Threat Ingestion API...")
        
        for i, threat in enumerate(SAMPLE_THREATS):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_base_url}/threats/ingest",
                    json=threat,
                    timeout=10
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ Threat {i+1} ingested successfully (ID: {result['id']}) - {response_time:.2f}ms")
                    self.test_results["ingestion"].append({
                        "threat_id": result['id'],
                        "status": "success",
                        "response_time_ms": response_time,
                        "classification": result.get('classification', {})
                    })
                else:
                    print(f"  ❌ Threat {i+1} ingestion failed: {response.status_code}")
                    self.test_results["ingestion"].append({
                        "status": "failed",
                        "error": response.text,
                        "response_time_ms": response_time
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Threat {i+1} ingestion error: {e}")
                self.test_results["ingestion"].append({
                    "status": "error",
                    "error": str(e)
                })
    
    async def test_bulk_ingestion(self):
        """Test bulk threat ingestion"""
        print("🧪 Testing Bulk Threat Ingestion...")
        
        # Create multiple variations of sample threats
        bulk_threats = []
        for _ in range(10):
            threat = random.choice(SAMPLE_THREATS).copy()
            threat["source_ip"] = f"192.168.1.{random.randint(1, 254)}"
            threat["severity"] = random.randint(1, 10)
            threat["confidence"] = random.uniform(0.1, 1.0)
            bulk_threats.append(threat)
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/threats/bulk-ingest",
                json=bulk_threats,
                timeout=30
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Bulk ingestion completed: {result['processed']} processed, {result['failed']} failed - {response_time:.2f}ms")
                self.test_results["performance"]["bulk_ingestion"] = {
                    "threats_count": len(bulk_threats),
                    "processed": result['processed'],
                    "failed": result['failed'],
                    "response_time_ms": response_time,
                    "throughput_per_second": len(bulk_threats) / (response_time / 1000)
                }
            else:
                print(f"  ❌ Bulk ingestion failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Bulk ingestion error: {e}")
    
    async def test_threat_retrieval(self):
        """Test threat retrieval APIs"""
        print("🧪 Testing Threat Retrieval APIs...")
        
        # Test global threats endpoint
        try:
            response = requests.get(f"{self.api_base_url}/threats/global?limit=100")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Global threats retrieved: {len(data['threats'])} threats")
                self.test_results["retrieval"].append({
                    "endpoint": "global",
                    "status": "success",
                    "count": len(data['threats'])
                })
            else:
                print(f"  ❌ Global threats retrieval failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Global threats retrieval error: {e}")
        
        # Test statistics endpoint
        try:
            response = requests.get(f"{self.api_base_url}/threats/statistics")
            if response.status_code == 200:
                stats = response.json()
                print(f"  ✅ Statistics retrieved: {stats['overview']['total_threats']} total threats")
                self.test_results["retrieval"].append({
                    "endpoint": "statistics",
                    "status": "success",
                    "data": stats['overview']
                })
            else:
                print(f"  ❌ Statistics retrieval failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Statistics retrieval error: {e}")
    
    async def test_health_check(self):
        """Test health check endpoint"""
        print("🧪 Testing Health Check...")
        
        try:
            response = requests.get(f"{self.api_base_url.replace('/api', '')}/api/health")
            if response.status_code == 200:
                health = response.json()
                print(f"  ✅ Health check passed: {health['status']}")
                print(f"    Database: {health.get('database', 'unknown')}")
                print(f"    WebSocket connections: {health.get('websocket_connections', 0)}")
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Health check error: {e}")
    
    async def test_performance(self):
        """Test system performance"""
        print("🧪 Testing Performance...")
        
        # Test concurrent ingestion
        concurrent_threats = [random.choice(SAMPLE_THREATS).copy() for _ in range(20)]
        for threat in concurrent_threats:
            threat["source_ip"] = f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        start_time = time.time()
        
        # Simulate concurrent requests
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for threat in concurrent_threats:
                future = executor.submit(
                    requests.post,
                    f"{self.api_base_url}/threats/ingest",
                    json=threat,
                    timeout=10
                )
                futures.append(future)
            
            successful = 0
            failed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    response = future.result()
                    if response.status_code == 200:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"  ✅ Concurrent ingestion: {successful} successful, {failed} failed")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Throughput: {successful / total_time:.2f} requests/second")
        
        self.test_results["performance"]["concurrent_ingestion"] = {
            "total_requests": len(concurrent_threats),
            "successful": successful,
            "failed": failed,
            "total_time_seconds": total_time,
            "throughput_per_second": successful / total_time
        }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 THREAT DETECTION SYSTEM TEST REPORT")
        print("="*60)
        
        # Ingestion results
        ingestion_success = len([r for r in self.test_results["ingestion"] if r.get("status") == "success"])
        ingestion_total = len(self.test_results["ingestion"])
        print(f"\n🔄 Threat Ingestion:")
        print(f"  Success Rate: {ingestion_success}/{ingestion_total} ({(ingestion_success/max(ingestion_total,1)*100):.1f}%)")
        
        if ingestion_success > 0:
            avg_response_time = sum(r.get("response_time_ms", 0) for r in self.test_results["ingestion"] if r.get("status") == "success") / ingestion_success
            print(f"  Average Response Time: {avg_response_time:.2f}ms")
        
        # Retrieval results
        retrieval_success = len([r for r in self.test_results["retrieval"] if r.get("status") == "success"])
        retrieval_total = len(self.test_results["retrieval"])
        print(f"\n📊 Data Retrieval:")
        print(f"  Success Rate: {retrieval_success}/{retrieval_total} ({(retrieval_success/max(retrieval_total,1)*100):.1f}%)")
        
        # Performance results
        if "concurrent_ingestion" in self.test_results["performance"]:
            perf = self.test_results["performance"]["concurrent_ingestion"]
            print(f"\n⚡ Performance:")
            print(f"  Concurrent Throughput: {perf['throughput_per_second']:.2f} requests/second")
            print(f"  Concurrent Success Rate: {perf['successful']}/{perf['total_requests']} ({(perf['successful']/perf['total_requests']*100):.1f}%)")
        
        if "bulk_ingestion" in self.test_results["performance"]:
            bulk = self.test_results["performance"]["bulk_ingestion"]
            print(f"  Bulk Throughput: {bulk['throughput_per_second']:.2f} threats/second")
        
        # Classification results
        classifications = [r.get("classification", {}) for r in self.test_results["ingestion"] if r.get("classification")]
        if classifications:
            print(f"\n🤖 ML Classification:")
            print(f"  Classifications Generated: {len(classifications)}")
            
            threat_types = [c.get("final_classification", {}).get("threat_type") for c in classifications]
            unique_types = set(filter(None, threat_types))
            print(f"  Unique Threat Types Detected: {len(unique_types)}")
            if unique_types:
                print(f"    Types: {', '.join(unique_types)}")
        
        print(f"\n✅ Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save detailed results
        with open(f"test_results_{int(time.time())}.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to test_results_{int(time.time())}.json")

async def main():
    """Run all tests"""
    print("🚀 Starting Threat Detection System Tests")
    print("="*60)
    
    tester = ThreatSystemTester()
    
    # Run all tests
    await tester.test_health_check()
    await tester.test_threat_ingestion()
    await tester.test_bulk_ingestion()
    await tester.test_threat_retrieval()
    await tester.test_performance()
    
    # Generate report
    tester.generate_test_report()

if __name__ == "__main__":
    asyncio.run(main())