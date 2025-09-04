#!/usr/bin/env python3
"""
GeoIP Service for Threat Location Resolution
Provides IP geolocation with caching and threat intelligence integration
"""

import geoip2.database
import geoip2.errors
import requests
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import ipaddress
import asyncio
import aiohttp
from dataclasses import dataclass
import os

@dataclass
class GeoLocation:
    """Geolocation data structure"""
    ip: str
    country: str
    country_code: str
    city: str
    latitude: float
    longitude: float
    timezone: str
    isp: str
    organization: str
    is_malicious: bool = False
    reputation_score: float = 0.5

class GeoIPService:
    """Advanced GeoIP service with multiple data sources"""
    
    def __init__(self, maxmind_db_path: str = None):
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.maxmind_db_path = maxmind_db_path or "GeoLite2-City.mmdb"
        self.maxmind_reader = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)

        # Threat intelligence APIs
        self.threat_apis = {
            'virustotal': {
                'url': 'https://www.virustotal.com/vtapi/v2/ip-address/report',
                'key': os.getenv('VIRUSTOTAL_API_KEY'),
                'enabled': bool(os.getenv('VIRUSTOTAL_API_KEY'))
            },
            'abuseipdb': {
                'url': 'https://api.abuseipdb.com/api/v2/check',
                'key': os.getenv('ABUSEIPDB_API_KEY'),
                'enabled': bool(os.getenv('ABUSEIPDB_API_KEY'))
            }
        }

        # Initialize MaxMind database
        self.init_maxmind_db()
    
    def init_maxmind_db(self):
        """Initialize MaxMind GeoIP database"""
        try:
            if os.path.exists(self.maxmind_db_path):
                self.maxmind_reader = geoip2.database.Reader(self.maxmind_db_path)
                self.logger.info(f"MaxMind database loaded: {self.maxmind_db_path}")
            else:
                self.logger.warning(f"MaxMind database not found: {self.maxmind_db_path}")
                self.logger.info("Download from: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data")
        except Exception as e:
            self.logger.error(f"Failed to initialize MaxMind database: {e}")
    
    async def get_location(self, ip_address: str, use_cache: bool = True) -> Optional[GeoLocation]:
        """Get comprehensive location data for IP address"""
        # Validate IP address
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            if ip_obj.is_private or ip_obj.is_loopback:
                return self._get_private_ip_location(ip_address)
        except ValueError:
            self.logger.error(f"Invalid IP address: {ip_address}")
            return None
        
        # Check cache first
        if use_cache and ip_address in self.cache:
            cached_data, timestamp = self.cache[ip_address]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        # Get location from multiple sources
        location_data = await self._get_location_from_sources(ip_address)
        
        if location_data:
            # Cache the result
            self.cache[ip_address] = (location_data, datetime.now())
            return location_data
        
        return None
    
    def _get_private_ip_location(self, ip_address: str) -> GeoLocation:
        """Handle private/internal IP addresses"""
        return GeoLocation(
            ip=ip_address,
            country="Internal Network",
            country_code="XX",
            city="Private Network",
            latitude=0.0,
            longitude=0.0,
            timezone="UTC",
            isp="Internal",
            organization="Private Network",
            is_malicious=False,
            reputation_score=1.0
        )
    
    async def _get_location_from_sources(self, ip_address: str) -> Optional[GeoLocation]:
        """Get location data from multiple sources"""
        location_data = None
        
        # Try MaxMind first (fastest)
        if self.maxmind_reader:
            location_data = self._get_maxmind_location(ip_address)
        
        # If MaxMind fails, try online services
        if not location_data:
            location_data = await self._get_online_location(ip_address)
        
        # Enhance with threat intelligence
        if location_data:
            threat_data = await self._get_threat_intelligence(ip_address)
            if threat_data:
                location_data.is_malicious = threat_data.get('is_malicious', False)
                location_data.reputation_score = threat_data.get('reputation_score', 0.5)
        
        return location_data
    
    def _get_maxmind_location(self, ip_address: str) -> Optional[GeoLocation]:
        """Get location from MaxMind database"""
        try:
            response = self.maxmind_reader.city(ip_address)
            
            return GeoLocation(
                ip=ip_address,
                country=response.country.name or "Unknown",
                country_code=response.country.iso_code or "XX",
                city=response.city.name or "Unknown",
                latitude=float(response.location.latitude or 0),
                longitude=float(response.location.longitude or 0),
                timezone=response.location.time_zone or "UTC",
                isp="Unknown",
                organization="Unknown"
            )
        except geoip2.errors.AddressNotFoundError:
            self.logger.warning(f"IP not found in MaxMind database: {ip_address}")
        except Exception as e:
            self.logger.error(f"MaxMind lookup error for {ip_address}: {e}")
        
        return None
    
    async def _get_online_location(self, ip_address: str) -> Optional[GeoLocation]:
        """Get location from online services"""
        # Try ip-api.com (free service)
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://ip-api.com/json/{ip_address}?fields=status,country,countryCode,city,lat,lon,timezone,isp,org"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'success':
                            return GeoLocation(
                                ip=ip_address,
                                country=data.get('country', 'Unknown'),
                                country_code=data.get('countryCode', 'XX'),
                                city=data.get('city', 'Unknown'),
                                latitude=float(data.get('lat', 0)),
                                longitude=float(data.get('lon', 0)),
                                timezone=data.get('timezone', 'UTC'),
                                isp=data.get('isp', 'Unknown'),
                                organization=data.get('org', 'Unknown')
                            )
        except Exception as e:
            self.logger.error(f"Online location lookup error for {ip_address}: {e}")
        
        return None
    
    async def _get_threat_intelligence(self, ip_address: str) -> Optional[Dict]:
        """Get threat intelligence for IP address"""
        threat_data = {
            'is_malicious': False,
            'reputation_score': 0.5,
            'threat_types': [],
            'last_seen': None
        }
        
        # Check VirusTotal
        if self.threat_apis['virustotal']['enabled']:
            vt_data = await self._check_virustotal(ip_address)
            if vt_data:
                threat_data.update(vt_data)
        
        # Check AbuseIPDB
        if self.threat_apis['abuseipdb']['enabled']:
            abuse_data = await self._check_abuseipdb(ip_address)
            if abuse_data:
                # Combine threat intelligence
                threat_data['is_malicious'] = threat_data['is_malicious'] or abuse_data.get('is_malicious', False)
                threat_data['reputation_score'] = min(
                    threat_data['reputation_score'],
                    abuse_data.get('reputation_score', 0.5)
                )
        
        return threat_data
    
    async def _check_virustotal(self, ip_address: str) -> Optional[Dict]:
        """Check IP against VirusTotal"""
        try:
            params = {
                'apikey': self.threat_apis['virustotal']['key'],
                'ip': ip_address
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.threat_apis['virustotal']['url'],
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('response_code') == 1:
                            positives = data.get('positives', 0)
                            total = data.get('total', 1)
                            
                            return {
                                'is_malicious': positives > 0,
                                'reputation_score': max(0, 1 - (positives / total)),
                                'threat_types': ['malware'] if positives > 0 else [],
                                'source': 'virustotal'
                            }
        except Exception as e:
            self.logger.error(f"VirusTotal check error for {ip_address}: {e}")
        
        return None
    
    async def _check_abuseipdb(self, ip_address: str) -> Optional[Dict]:
        """Check IP against AbuseIPDB"""
        try:
            headers = {
                'Key': self.threat_apis['abuseipdb']['key'],
                'Accept': 'application/json'
            }
            params = {
                'ipAddress': ip_address,
                'maxAgeInDays': 90,
                'verbose': ''
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.threat_apis['abuseipdb']['url'],
                    headers=headers,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data:
                            abuse_confidence = data['data'].get('abuseConfidencePercentage', 0)
                            
                            return {
                                'is_malicious': abuse_confidence > 25,
                                'reputation_score': max(0, 1 - (abuse_confidence / 100)),
                                'threat_types': ['abuse'] if abuse_confidence > 25 else [],
                                'source': 'abuseipdb'
                            }
        except Exception as e:
            self.logger.error(f"AbuseIPDB check error for {ip_address}: {e}")
        
        return None
    
    async def bulk_lookup(self, ip_addresses: list) -> Dict[str, GeoLocation]:
        """Perform bulk IP lookups"""
        results = {}
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(ip_addresses), batch_size):
            batch = ip_addresses[i:i + batch_size]
            
            tasks = [self.get_location(ip) for ip in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ip, result in zip(batch, batch_results):
                if isinstance(result, GeoLocation):
                    results[ip] = result
                elif isinstance(result, Exception):
                    self.logger.error(f"Error processing {ip}: {result}")
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        return results
    
    def get_country_threat_stats(self, threats: list) -> Dict[str, Dict]:
        """Generate country-based threat statistics"""
        country_stats = {}
        
        for threat in threats:
            country_code = threat.get('source_country_code', 'XX')
            if country_code not in country_stats:
                country_stats[country_code] = {
                    'country': threat.get('source_country', 'Unknown'),
                    'threat_count': 0,
                    'severity_sum': 0,
                    'avg_severity': 0,
                    'threat_types': set(),
                    'coordinates': [
                        threat.get('source_latitude', 0),
                        threat.get('source_longitude', 0)
                    ]
                }
            
            stats = country_stats[country_code]
            stats['threat_count'] += 1
            stats['severity_sum'] += threat.get('severity', 1)
            stats['avg_severity'] = stats['severity_sum'] / stats['threat_count']
            stats['threat_types'].add(threat.get('threat_type', 'unknown'))
        
        # Convert sets to lists for JSON serialization
        for country_code in country_stats:
            country_stats[country_code]['threat_types'] = list(
                country_stats[country_code]['threat_types']
            )
        
        return country_stats
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        now = datetime.now()
        expired_keys = [
            ip for ip, (_, timestamp) in self.cache.items()
            if now - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Global instance
geoip_service = GeoIPService()

if __name__ == "__main__":
    # Test the service
    async def test_service():
        test_ips = ["8.8.8.8", "1.1.1.1", "192.168.1.1"]
        
        for ip in test_ips:
            location = await geoip_service.get_location(ip)
            if location:
                print(f"{ip}: {location.city}, {location.country} ({location.latitude}, {location.longitude})")
            else:
                print(f"{ip}: Location not found")
    
    asyncio.run(test_service())