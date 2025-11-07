"""
Air Quality Index (AQI) Fetcher Module
Handles fetching and processing air quality data
"""

import requests
import os
from typing import Dict, Tuple
from datetime import datetime
from loguru import logger
import numpy as np


def fetch_air_quality(lat: float, lon: float) -> Dict:
    """
    Fetch air quality data from OpenWeatherMap Air Pollution API
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with AQI and pollutant data
    """
    logger.info(f"Fetching air quality data for location ({lat}, {lon})")
    
    # OpenWeatherMap Air Pollution API endpoint
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        logger.warning("OpenWeatherMap API key not configured, returning dummy AQI data")
        return _get_dummy_aqi_data((lat, lon))
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'list' in data and len(data['list']) > 0:
            aqi_data = data['list'][0]
            
            # Map AQI index to descriptive levels
            aqi_levels = {
                1: 'Good',
                2: 'Fair',
                3: 'Moderate',
                4: 'Poor',
                5: 'Very Poor'
            }
            
            aqi_index = aqi_data['main']['aqi']
            components = aqi_data['components']
            
            result = {
                'date': datetime.now().isoformat(),
                'location': {'latitude': lat, 'longitude': lon},
                'aqi': aqi_index,
                'aqi_level': aqi_levels.get(aqi_index, 'Unknown'),
                'pollutants': {
                    'pm2_5': components.get('pm2_5', 0),  # Fine particles (μg/m³)
                    'pm10': components.get('pm10', 0),     # Coarse particles (μg/m³)
                    'o3': components.get('o3', 0),         # Ozone (μg/m³)
                    'no2': components.get('no2', 0),       # Nitrogen dioxide (μg/m³)
                    'so2': components.get('so2', 0),       # Sulfur dioxide (μg/m³)
                    'co': components.get('co', 0)          # Carbon monoxide (μg/m³)
                },
                'crop_impact': _assess_crop_impact(aqi_index, components),
                'source': 'OpenWeatherMap Air Pollution API'
            }
            
            logger.info(f"Successfully fetched AQI data: {aqi_levels.get(aqi_index)} (Index: {aqi_index})")
            return result
        else:
            logger.warning("No AQI data available from API")
            return _get_dummy_aqi_data((lat, lon))
            
    except Exception as e:
        logger.error(f"Error fetching AQI data: {e}")
        return _get_dummy_aqi_data((lat, lon))


def _assess_crop_impact(aqi_index: int, components: Dict) -> Dict:
    """Assess the impact of air quality on crops"""
    pm25 = components.get('pm2_5', 0)
    o3 = components.get('o3', 0)
    no2 = components.get('no2', 0)
    so2 = components.get('so2', 0)
    
    impacts = []
    severity = 'low'
    
    # PM2.5 impact (fine particulate matter)
    if pm25 > 75:
        impacts.append("High PM2.5: Reduces photosynthesis by blocking sunlight")
        severity = 'high'
    elif pm25 > 35:
        impacts.append("Moderate PM2.5: May reduce plant growth")
        severity = 'moderate' if severity != 'high' else severity
    
    # Ozone impact
    if o3 > 120:
        impacts.append("High O₃: Causes leaf damage, reduces crop yield by 10-30%")
        severity = 'high'
    elif o3 > 80:
        impacts.append("Moderate O₃: May cause leaf stippling and reduced growth")
        severity = 'moderate' if severity != 'high' else severity
    
    # NO2 and SO2 (acid rain precursors)
    if no2 > 200 or so2 > 350:
        impacts.append("High NO₂/SO₂: Risk of acid rain, leaf damage")
        severity = 'high'
    
    if not impacts:
        impacts.append("Air quality supports healthy crop growth")
    
    recommendations = _get_aqi_recommendations(severity)
    
    return {
        'severity': severity,
        'impacts': impacts,
        'recommendations': recommendations
    }


def _get_aqi_recommendations(severity: str) -> list:
    """Get farming recommendations based on AQI severity"""
    if severity == 'high':
        return [
            "Consider delaying field operations during high pollution hours",
            "Increase irrigation to help plants cope with stress",
            "Monitor crops for visible leaf damage (stippling, bronzing)",
            "Consider using anti-transpirant sprays to reduce ozone uptake"
        ]
    elif severity == 'moderate':
        return [
            "Monitor crop health for stress symptoms",
            "Maintain adequate soil moisture"
        ]
    else:
        return ["Continue normal farming operations"]


def _get_dummy_aqi_data(location: Tuple[float, float]) -> Dict:
    """Generate realistic dummy AQI data for testing"""
    lat, lon = location
    
    # Estimate based on region (India/Asia typically has higher pollution)
    if 5 <= lat <= 35 and 60 <= lon <= 100:  # India/South Asia region
        aqi_index = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        pm25 = np.random.uniform(35, 150)
        pm10 = np.random.uniform(50, 250)
        o3 = np.random.uniform(60, 150)
    else:
        aqi_index = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        pm25 = np.random.uniform(5, 35)
        pm10 = np.random.uniform(10, 50)
        o3 = np.random.uniform(30, 80)
    
    aqi_levels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
    
    components = {
        'pm2_5': pm25,
        'pm10': pm10,
        'o3': o3,
        'no2': np.random.uniform(10, 100),
        'so2': np.random.uniform(5, 50),
        'co': np.random.uniform(200, 800)
    }
    
    return {
        'date': datetime.now().isoformat(),
        'location': {'latitude': lat, 'longitude': lon},
        'aqi': aqi_index,
        'aqi_level': aqi_levels[aqi_index],
        'pollutants': components,
        'crop_impact': _assess_crop_impact(aqi_index, components),
        'source': 'Estimated (API unavailable)'
    }
