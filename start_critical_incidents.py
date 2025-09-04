#!/usr/bin/env python3
"""
Startup script for Critical Incidents Detection Backend
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'flask-socketio', 
        'flask-cors',
        'supabase',
        'python-dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   {package}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True

def check_environment():
    """Check if environment variables are set"""
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   {var}")
        print("\nPlease set these in your .env file")
        return False
    
    return True

def main():
    print("🚨 Starting Critical Incidents Detection Backend")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Start the backend
    print("✅ All dependencies and environment variables are ready")
    print("🚀 Starting critical incidents backend on port 8002...")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Run the critical incidents backend
        subprocess.run([sys.executable, 'critical_incidents_backend.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Critical incidents backend stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ critical_incidents_backend.py not found")
        print("   Make sure you're running this from the project root directory")
        sys.exit(1)

if __name__ == "__main__":
    main()