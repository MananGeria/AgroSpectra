"""
Tier 2: Data Acquisition Layer
Manages automated retrieval from Sentinel Hub, OpenWeatherMap, and MOSDAC APIs
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    SentinelHubCatalog,
)
from pyowm import OWM
from loguru import logger
import yaml


class FetchController:
    """Main controller for data acquisition from multiple sources"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize fetch controller with configuration"""
        self.config = self._load_config(config_path)
        self.cache_raw = Path(self.config['cache']['raw_data']['path'])
        self.cache_raw.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients
        self._init_sentinel_hub()
        self._init_openweathermap()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_sentinel_hub(self):
        """Initialize Sentinel Hub API client"""
        sh_config = self.config['sentinel_hub']
        self.sh_config = SHConfig()
        # Try both naming conventions
        self.sh_config.sh_client_id = os.getenv('SENTINEL_CLIENT_ID') or os.getenv('SENTINEL_HUB_CLIENT_ID', sh_config['client_id'])
        self.sh_config.sh_client_secret = os.getenv('SENTINEL_CLIENT_SECRET') or os.getenv('SENTINEL_HUB_CLIENT_SECRET', sh_config['client_secret'])
        self.sh_config.instance_id = os.getenv('SENTINEL_INSTANCE_ID') or os.getenv('SENTINEL_HUB_INSTANCE_ID', sh_config['instance_id'])
        
        if not all([self.sh_config.sh_client_id, self.sh_config.sh_client_secret]):
            logger.warning("Sentinel Hub credentials not configured")
    
    def _init_openweathermap(self):
        """Initialize OpenWeatherMap API client"""
        api_key = os.getenv('OPENWEATHER_API_KEY', 
                           self.config['openweathermap']['api_key'])
        if api_key and api_key != "${OPENWEATHER_API_KEY}":
            self.owm = OWM(api_key)
            self.weather_mgr = self.owm.weather_manager()
        else:
            logger.warning("OpenWeatherMap API key not configured")
            self.owm = None
    
    def fetch_sentinel_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str = None,
        end_date: str = None,
        date_range: Tuple[str, str] = None,
        output_dir: Path = None,
        bands: List[str] = ['B03', 'B04', 'B08'],
        resolution: int = 10,
        max_cloud_cover: float = 20.0
    ) -> Dict:
        """
        Fetch Sentinel-2 imagery for specified area and time period
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            date_range: Alternative: Date range tuple (start_date, end_date)
            output_dir: Directory to save output (optional)
            bands: List of Sentinel-2 bands to retrieve
            resolution: Spatial resolution in meters
            max_cloud_cover: Maximum cloud coverage percentage
            
        Returns:
            Dictionary with imagery data and metadata
        """
        # Handle both parameter styles
        if date_range is None:
            date_range = (start_date, end_date)
        
        logger.info(f"Fetching Sentinel-2 imagery for bbox {bbox}, dates {date_range}")
        
        # Create bounding box
        bbox_obj = BBox(bbox=bbox, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox_obj, resolution=resolution)
        
        # Create evalscript for band retrieval
        evalscript = self._create_evalscript(bands)
        
        # Create request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=date_range,
                    maxcc=max_cloud_cover / 100.0,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF),
                SentinelHubRequest.output_response('SCL', MimeType.TIFF),  # Scene Classification
            ],
            bbox=bbox_obj,
            size=size,
            config=self.sh_config,
        )
        
        try:
            # Fetch data
            data = request.get_data()
            
            # Save to cache
            cache_file = self._save_sentinel_cache(data, bbox, date_range, bands)
            
            # Create metadata
            metadata = self._create_sentinel_metadata(
                bbox, date_range, bands, resolution, cache_file
            )
            
            logger.info(f"Successfully fetched Sentinel-2 data: {cache_file}")
            
            return {
                'data': data,
                'metadata': metadata,
                'cache_file': cache_file
            }
            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-2 data: {str(e)}")
            return None
    
    def _create_evalscript(self, bands: List[str]) -> str:
        """Create evalscript for Sentinel Hub API"""
        # Quote band names properly for JavaScript
        band_list = ', '.join([f'"{b}"' for b in bands])
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{
                    bands: [{band_list}, "SCL"],
                    units: "REFLECTANCE"
                }}],
                output: [
                    {{
                        id: "default",
                        bands: {len(bands)},
                        sampleType: "FLOAT32"
                    }},
                    {{
                        id: "SCL",
                        bands: 1,
                        sampleType: "UINT8"
                    }}
                ]
            }};
        }}
        
        function evaluatePixel(sample) {{
            return {{
                default: [{', '.join([f'sample.{b}' for b in bands])}],
                SCL: [sample.SCL]
            }};
        }}
        """
    
    def _save_sentinel_cache(
        self, 
        data: List, 
        bbox: Tuple, 
        date_range: Tuple, 
        bands: List[str]
    ) -> str:
        """Save Sentinel data to cache"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = self.cache_raw / 'sentinel' / date_range[0]
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"sentinel2_{timestamp}.tif"
        
        # Save imagery (data[0] contains the imagery)
        with open(cache_file, 'wb') as f:
            f.write(data[0])
        
        # Save scene classification layer
        scl_file = cache_dir / f"sentinel2_{timestamp}_SCL.tif"
        with open(scl_file, 'wb') as f:
            f.write(data[1])
        
        return str(cache_file)
    
    def _create_sentinel_metadata(
        self,
        bbox: Tuple,
        date_range: Tuple,
        bands: List[str],
        resolution: int,
        cache_file: str
    ) -> Dict:
        """Create metadata for Sentinel imagery"""
        return {
            'source': 'Sentinel-2 L2A',
            'acquisition_time': datetime.now().isoformat(),
            'bbox': bbox,
            'crs': 'EPSG:4326',
            'date_range': date_range,
            'bands': bands,
            'resolution': resolution,
            'cache_file': cache_file,
            'processor': 'SentinelHub API',
            'version': '1.0'
        }
    
    def fetch_weather_data(
        self,
        lat: float = None,
        lon: float = None,
        location: Tuple[float, float] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        date: Optional[str] = None
    ) -> Dict:
        """
        Fetch weather data from OpenWeatherMap
        
        Args:
            lat: Latitude (alternative to location)
            lon: Longitude (alternative to location)
            location: (latitude, longitude) tuple
            start_date: Start date for historical data
            end_date: End date for historical data
            date: Date in 'YYYY-MM-DD' format (None for current weather)
            
        Returns:
            Dictionary with weather data
        """
        # Handle both parameter styles
        if location is None:
            if lat is not None and lon is not None:
                location = (lat, lon)
            else:
                raise ValueError("Must provide either location tuple or lat/lon")
        
        if not self.owm:
            logger.warning("OpenWeatherMap not configured, returning dummy data")
            return self._get_dummy_weather_data(location, date)
        
        lat, lon = location
        logger.info(f"Fetching weather data for location ({lat}, {lon})")
        
        try:
            if date is None:
                # Current weather
                observation = self.weather_mgr.weather_at_coords(lat, lon)
                weather = observation.weather
                
                data = {
                    'date': datetime.now().isoformat(),
                    'location': {'latitude': lat, 'longitude': lon},
                    'temperature': weather.temperature('celsius')['temp'],
                    'humidity': weather.humidity,
                    'rainfall': weather.rain.get('1h', 0) if weather.rain else 0,
                    'wind_speed': weather.wind()['speed'],
                    'cloud_cover': weather.clouds,
                    'status': weather.status,
                    'detailed_status': weather.detailed_status
                }
            else:
                # Historical weather (requires paid API)
                logger.warning("Historical weather data requires paid OpenWeatherMap plan")
                data = self._get_dummy_weather_data(location, date)
            
            # Save to cache
            cache_file = self._save_weather_cache(data, location, date)
            data['cache_file'] = cache_file
            
            logger.info("Successfully fetched weather data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return self._get_dummy_weather_data(location, date)
    
    def _get_dummy_weather_data(self, location: Tuple[float, float], date: Optional[str]) -> Dict:
        """Generate dummy weather data for testing"""
        return {
            'date': date or datetime.now().isoformat(),
            'location': {'latitude': location[0], 'longitude': location[1]},
            'temperature': 28.5,
            'humidity': 75.0,
            'rainfall': 0.0,
            'wind_speed': 3.5,
            'cloud_cover': 20,
            'status': 'Clear',
            'detailed_status': 'clear sky',
            'note': 'Dummy data - Configure OpenWeatherMap API key for real data'
        }
    
    def _save_weather_cache(
        self,
        data: Dict,
        location: Tuple[float, float],
        date: Optional[str]
    ) -> str:
        """Save weather data to cache"""
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        cache_dir = self.cache_raw / 'weather' / date_str
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = cache_dir / f"weather_{timestamp}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(cache_file)
    
    def fetch_mosdac_data(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        products: List[str] = ['land_surface_temperature']
    ) -> Dict:
        """
        Fetch data from MOSDAC portal
        
        Note: This is a placeholder. Actual MOSDAC integration requires
        manual download or specific API access approval
        
        Args:
            bbox: Bounding box
            date_range: Date range
            products: List of products to fetch
            
        Returns:
            Dictionary with MOSDAC data information
        """
        logger.warning("MOSDAC integration requires manual setup. Using placeholder.")
        
        return {
            'source': 'MOSDAC',
            'bbox': bbox,
            'date_range': date_range,
            'products': products,
            'status': 'manual_download_required',
            'instructions': [
                '1. Visit https://www.mosdac.gov.in/',
                '2. Register and login',
                '3. Navigate to data products section',
                '4. Select products: ' + ', '.join(products),
                '5. Specify spatial and temporal parameters',
                '6. Download NetCDF files',
                f"7. Place files in {self.cache_raw / 'mosdac'}"
            ]
        }
    
    def fetch_soil_data(
        self,
        bbox: Tuple[float, float, float, float],
        sources: List[str] = ['icar', 'nbss_lup', 'bhuvan']
    ) -> Dict:
        """
        Fetch soil data from configured databases
        
        Note: This is a placeholder. Actual soil database integration
        requires specific data files or API access
        
        Args:
            bbox: Bounding box
            sources: List of soil data sources
            
        Returns:
            Dictionary with soil data information
        """
        logger.warning("Soil database integration requires manual setup. Using placeholder.")
        
        return {
            'source': 'Soil Databases',
            'bbox': bbox,
            'sources': sources,
            'status': 'manual_setup_required',
            'instructions': [
                'ICAR: Download district-level soil data from https://krishi.icar.gov.in/',
                'NBSS & LUP: Request data from https://www.nbsslup.in/',
                'Bhuvan: Access via https://bhuvan.nrsc.gov.in/',
                f"Place shapefiles/CSV in {self.cache_raw / 'soil'}"
            ],
            'expected_attributes': [
                'nitrogen', 'phosphorus', 'potassium',
                'ph', 'texture', 'organic_carbon'
            ]
        }


class MetaLogger:
    """Metadata logging for all acquired datasets"""
    
    def __init__(self, base_path: str = "data/cache/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def log_metadata(self, metadata: Dict, data_type: str):
        """
        Log metadata for acquired dataset
        
        Args:
            metadata: Metadata dictionary
            data_type: Type of data (sentinel, weather, mosdac, soil)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_dir = self.base_path / 'metadata' / data_type
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        meta_file = meta_dir / f"metadata_{timestamp}.json"
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata logged: {meta_file}")
        return str(meta_file)
    
    def fetch_air_quality(
        self,
        lat: float = None,
        lon: float = None,
        location: Tuple[float, float] = None
    ) -> Dict:
        """
        Fetch air quality data from OpenWeatherMap Air Pollution API
        
        Args:
            lat: Latitude (alternative to location)
            lon: Longitude (alternative to location)
            location: (latitude, longitude) tuple
            
        Returns:
            Dictionary with AQI and pollutant data
        """
        # Handle both parameter styles
        if location is None:
            if lat is not None and lon is not None:
                location = (lat, lon)
            else:
                raise ValueError("Must provide either location tuple or lat/lon")
        
        lat, lon = location
        logger.info(f"Fetching air quality data for location ({lat}, {lon})")
        
        # OpenWeatherMap Air Pollution API endpoint
        api_key = os.getenv('OPENWEATHER_API_KEY')
        
        if not api_key:
            logger.warning("OpenWeatherMap API key not configured, returning dummy AQI data")
            return self._get_dummy_aqi_data(location)
        
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
                    'crop_impact': self._assess_crop_impact(aqi_index, components),
                    'source': 'OpenWeatherMap Air Pollution API'
                }
                
                logger.info(f"Successfully fetched AQI data: {aqi_levels.get(aqi_index)} (Index: {aqi_index})")
                return result
            else:
                logger.warning("No AQI data available from API")
                return self._get_dummy_aqi_data(location)
                
        except Exception as e:
            logger.error(f"Error fetching AQI data: {e}")
            return self._get_dummy_aqi_data(location)
    
    def _assess_crop_impact(self, aqi_index: int, components: dict) -> Dict:
        """Assess the impact of air quality on crops"""
        pm25 = components.get('pm2_5', 0)
        pm10 = components.get('pm10', 0)
        o3 = components.get('o3', 0)
        no2 = components.get('no2', 0)
        so2 = components.get('so2', 0)
        
        impacts = []
        severity = 'low'
        
        # PM2.5 impact (blocks sunlight, deposits on leaves)
        if pm25 > 75:
            severity = 'high'
            impacts.append('⚠️ High PM2.5: Reduces photosynthesis by blocking sunlight')
        elif pm25 > 35:
            severity = 'moderate'
            impacts.append('⚠️ Elevated PM2.5: May reduce photosynthesis')
        
        # Ozone impact (damages leaves, reduces yield)
        if o3 > 120:
            severity = 'high'
            impacts.append('🔴 High Ozone: Causes leaf damage, reduces crop yield by 10-30%')
        elif o3 > 80:
            severity = 'moderate'
            impacts.append('⚠️ Elevated Ozone: May cause leaf damage on sensitive crops')
        
        # NO2 and SO2 (acid rain precursors)
        if no2 > 200 or so2 > 350:
            severity = 'high'
            impacts.append('⚠️ High NO₂/SO₂: Risk of acid rain, leaf damage')
        
        if not impacts:
            impacts.append('✅ Good air quality - minimal crop impact')
            severity = 'low'
        
        return {
            'severity': severity,
            'impacts': impacts,
            'recommendations': self._get_aqi_recommendations(aqi_index, severity)
        }
    
    def _get_aqi_recommendations(self, aqi_index: int, severity: str) -> list:
        """Get recommendations based on AQI"""
        if severity == 'high':
            return [
                'Consider delaying field operations during high pollution hours',
                'Increase irrigation to help plants cope with stress',
                'Monitor crops for visible leaf damage (stippling, bronzing)',
                'Consider using anti-transpirant sprays to reduce ozone uptake',
                'Harvest sensitive crops early if prolonged poor AQI expected'
            ]
        elif severity == 'moderate':
            return [
                'Monitor crop health for stress symptoms',
                'Maintain adequate soil moisture',
                'Avoid excessive fertilization (may increase ozone sensitivity)'
            ]
        else:
            return [
                'Continue normal farming operations',
                'Good air quality supports healthy crop growth'
            ]
    
    def _get_dummy_aqi_data(self, location: Tuple[float, float]) -> Dict:
        """Generate dummy AQI data when API is unavailable"""
        import random
        
        # Generate realistic AQI based on region (rough estimates)
        lat, lon = location
        
        # India/Asia tends to have higher pollution
        if 5 < lat < 35 and 60 < lon < 100:
            aqi_index = random.choice([3, 3, 4, 4, 4, 5])  # Moderate to Very Poor
            pm25 = random.uniform(35, 150)
            pm10 = random.uniform(50, 200)
        else:
            aqi_index = random.choice([1, 2, 2, 3, 3])  # Good to Moderate
            pm25 = random.uniform(5, 35)
            pm10 = random.uniform(10, 50)
        
        aqi_levels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
        
        components = {
            'pm2_5': pm25,
            'pm10': pm10,
            'o3': random.uniform(40, 100),
            'no2': random.uniform(10, 80),
            'so2': random.uniform(5, 50),
            'co': random.uniform(200, 800)
        }
        
        return {
            'date': datetime.now().isoformat(),
            'location': {'latitude': lat, 'longitude': lon},
            'aqi': aqi_index,
            'aqi_level': aqi_levels[aqi_index],
            'pollutants': components,
            'crop_impact': self._assess_crop_impact(aqi_index, components),
            'source': 'Simulated data (API unavailable)',
            'note': 'Using estimated values based on region'
        }


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize fetch controller
    fetcher = FetchController()
    
    # Example: Fetch Sentinel-2 imagery
    bbox = (77.5, 28.4, 77.7, 28.6)  # Delhi region
    date_range = ('2024-06-01', '2024-06-15')
    
    # sentinel_data = fetcher.fetch_sentinel_imagery(bbox, date_range)
    
    # Example: Fetch weather data
    location = (28.5, 77.6)
    weather_data = fetcher.fetch_weather_data(location)
    
    print("Weather data:", json.dumps(weather_data, indent=2))
