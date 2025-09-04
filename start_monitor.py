#!/usr/bin/env python3
"""
Network Monitor Startup Script
Starts network traffic monitoring on multiple interfaces
"""

import sys
import threading
import time
from network_traffic_monitor import NetworkTrafficMonitor

def get_available_interfaces():
    """Get list of available network interfaces"""
    import psutil
    interfaces = []
    
    # Get network interface statistics
    net_if_stats = psutil.net_if_stats()
    
    for interface_name, stats in net_if_stats.items():
        # Skip loopback and inactive interfaces
        if interface_name != 'lo' and stats.isup:
            interfaces.append(interface_name)
    
    return interfaces

def monitor_interface(interface_name, max_packets=1000):
    """Monitor a specific network interface"""
    print(f"🔍 Starting monitor for interface: {interface_name}")
    try:
        monitor = NetworkTrafficMonitor(interface=interface_name, max_packets=max_packets)
        monitor.start_monitoring(display_interval=15)  # Update every 15 seconds
    except Exception as e:
        print(f"❌ Error monitoring {interface_name}: {e}")

def main():
    print("🛡️  Multi-Interface Network Security Monitor")
    print("=" * 50)
    
    # Get available interfaces
    available_interfaces = get_available_interfaces()
    
    if not available_interfaces:
        print("❌ No active network interfaces found")
        sys.exit(1)
    
    print(f"📡 Found {len(available_interfaces)} active interfaces:")
    for i, interface in enumerate(available_interfaces, 1):
        print(f"   {i}. {interface}")
    
    # Common interface names to prioritize
    priority_interfaces = ['eth0', 'eth1', 'en0', 'en1', 'wlan0', 'wlan1', 'ens33', 'enp0s3']
    
    # Select interfaces to monitor (prioritize common ones)
    interfaces_to_monitor = []
    
    # Add priority interfaces if they exist
    for priority in priority_interfaces:
        if priority in available_interfaces:
            interfaces_to_monitor.append(priority)
    
    # If no priority interfaces found, use first 2 available
    if not interfaces_to_monitor:
        interfaces_to_monitor = available_interfaces[:2]
    
    # Limit to 2 interfaces to avoid overwhelming the system
    interfaces_to_monitor = interfaces_to_monitor[:2]
    
    print(f"\n🚀 Starting monitoring on {len(interfaces_to_monitor)} interface(s):")
    for interface in interfaces_to_monitor:
        print(f"   • {interface}")
    
    print("\n⚠️  Note: Requires administrator/root privileges for packet capture")
    print("🔄 Each interface will update every 15 seconds")
    print("🛑 Press Ctrl+C to stop all monitors\n")
    
    # Start monitoring threads for each interface
    monitor_threads = []
    
    for interface in interfaces_to_monitor:
        thread = threading.Thread(
            target=monitor_interface,
            args=(interface, 1500),  # 1500 packets per interface
            daemon=True
        )
        thread.start()
        monitor_threads.append(thread)
        time.sleep(1)  # Stagger startup to avoid conflicts
    
    try:
        # Keep main thread alive
        print("✅ All monitors started successfully")
        print("📊 Monitoring network traffic on multiple interfaces...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all network monitors...")
        print("📄 Security reports will be saved for each interface")
        
        # Wait for threads to finish gracefully
        for thread in monitor_threads:
            thread.join(timeout=5)
        
        print("✅ All monitors stopped")

if __name__ == "__main__":
    main()
