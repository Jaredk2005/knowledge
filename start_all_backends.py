#!/usr/bin/env python3
"""
Startup script to run all backend services
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def run_service(script_name, port, service_name):
    """Run a backend service"""
    try:
        print(f"🚀 Starting {service_name} on port {port}...")
        subprocess.run([sys.executable, script_name], check=True)
    except KeyboardInterrupt:
        print(f"\n🛑 {service_name} stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting {service_name}: {e}")
    except FileNotFoundError:
        print(f"❌ {script_name} not found")

def main():
    print("🛡️  Starting All Cybersecurity Backend Services")
    print("=" * 60)
    
    services = [
        {
            'script': 'api_server.py',
            'port': 8000,
            'name': 'Main API Server'
        },
        {
            'script': 'backend/threat_ingestion_api.py',
            'port': 8001,
            'name': 'Threat Ingestion API'
        },
        {
            'script': 'critical_incidents_backend.py',
            'port': 8002,
            'name': 'Critical Incidents Backend'
        },
        {
            'script': 'threat_detection_backend.py',
            'port': 5000,
            'name': 'Threat Detection Backend'
        }
    ]
    
    print("Services to start:")
    for service in services:
        print(f"  • {service['name']} (port {service['port']})")
    print()
    
    # Check if files exist
    missing_files = []
    for service in services:
        if not os.path.exists(service['script']):
            missing_files.append(service['script'])
    
    if missing_files:
        print("❌ Missing backend files:")
        for file in missing_files:
            print(f"   {file}")
        print("\nPlease ensure all backend files are present")
        sys.exit(1)
    
    print("✅ All backend files found")
    print("🚀 Starting services...")
    print("   Press Ctrl+C to stop all services")
    print()
    
    # Start services in separate threads
    threads = []
    for service in services:
        thread = threading.Thread(
            target=run_service,
            args=(service['script'], service['port'], service['name']),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        time.sleep(2)  # Stagger startup
    
    try:
        # Wait for all threads
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all backend services...")

if __name__ == "__main__":
    main()