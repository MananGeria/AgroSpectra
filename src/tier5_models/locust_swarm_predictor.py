"""
Locust Swarm Risk Prediction Model
Predicts desert locust swarm probability based on weather, vegetation, and historical data
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from loguru import logger


class LocustSwarmPredictor:
    """
    Predicts locust swarm risk using meteorological and vegetation data
    
    Key Factors:
    - Temperature: 25-35°C optimal for breeding
    - Rainfall: Recent rains (10-30mm) trigger breeding
    - Vegetation: NDVI > 0.4 indicates suitable food
    - Humidity: >60% supports egg development
    - Wind: Swarms travel 100-200 km/day
    - Season: Peak during monsoon and post-monsoon
    """
    
    def __init__(self):
        # Historical locust-prone regions (lat, lon radius in degrees)
        self.high_risk_zones = [
            (25.0, 70.0, 5.0),  # Rajasthan, India
            (28.0, 68.0, 4.0),  # Sindh, Pakistan
            (15.0, 45.0, 10.0), # Yemen/Somalia
            (20.0, 55.0, 8.0),  # Arabian Peninsula
        ]
        
        # Season-based risk multipliers (month: multiplier)
        self.seasonal_risk = {
            1: 0.3,  # January - Low
            2: 0.4,  # February - Low-Med
            3: 0.6,  # March - Medium
            4: 0.8,  # April - Med-High
            5: 1.0,  # May - High (breeding)
            6: 1.0,  # June - High (breeding)
            7: 0.9,  # July - High
            8: 0.8,  # August - Med-High
            9: 0.7,  # September - Medium
            10: 0.5, # October - Low-Med
            11: 0.4, # November - Low
            12: 0.3, # December - Low
        }
    
    def predict_swarm_risk(
        self,
        lat: float,
        lon: float,
        temperature: float,
        humidity: float,
        rainfall_15days: float,
        ndvi: float,
        wind_speed: float = None,
        date: datetime = None
    ) -> Dict:
        """
        Predict locust swarm risk for a location
        
        Args:
            lat: Latitude
            lon: Longitude
            temperature: Average temperature (°C)
            humidity: Average humidity (%)
            rainfall_15days: Cumulative rainfall in last 15 days (mm)
            ndvi: Vegetation index (0-1)
            wind_speed: Wind speed (km/h), optional
            date: Date for prediction, defaults to now
            
        Returns:
            Dict with risk score, category, and explanations
        """
        if date is None:
            date = datetime.now()
        
        # 1. Geographic proximity to known breeding areas
        geo_risk = self._calculate_geographic_risk(lat, lon)
        
        # 2. Temperature suitability (optimal: 25-35°C)
        temp_risk = self._calculate_temperature_risk(temperature)
        
        # 3. Rainfall trigger (10-30mm in 2 weeks = high risk)
        rain_risk = self._calculate_rainfall_risk(rainfall_15days)
        
        # 4. Vegetation availability (NDVI > 0.4 = food available)
        veg_risk = self._calculate_vegetation_risk(ndvi)
        
        # 5. Humidity for egg survival (>60% = favorable)
        humidity_risk = self._calculate_humidity_risk(humidity)
        
        # 6. Seasonal factor
        seasonal_mult = self.seasonal_risk.get(date.month, 0.5)
        
        # 7. Wind speed (if available)
        wind_risk = self._calculate_wind_risk(wind_speed) if wind_speed else 0.5
        
        # Weighted combination
        base_risk = (
            geo_risk * 0.25 +
            temp_risk * 0.20 +
            rain_risk * 0.20 +
            veg_risk * 0.15 +
            humidity_risk * 0.10 +
            wind_risk * 0.10
        )
        
        # Apply seasonal multiplier
        final_risk = base_risk * seasonal_mult
        
        # Ensure 0-1 range
        final_risk = np.clip(final_risk, 0.0, 1.0)
        
        # Categorize risk
        category, emoji = self._categorize_risk(final_risk)
        
        # Generate explanation
        factors = self._explain_factors(
            geo_risk, temp_risk, rain_risk, veg_risk, 
            humidity_risk, wind_risk, seasonal_mult
        )
        
        # Recommendations
        recommendations = self._generate_recommendations(final_risk, factors)
        
        return {
            'risk_score': round(final_risk, 3),
            'risk_percentage': round(final_risk * 100, 1),
            'category': category,
            'emoji': emoji,
            'factors': factors,
            'recommendations': recommendations,
            'season': date.strftime('%B'),
            'seasonal_multiplier': seasonal_mult,
            'next_inspection_days': self._calculate_inspection_interval(final_risk)
        }
    
    def _calculate_geographic_risk(self, lat: float, lon: float) -> float:
        """Check if location is near known locust breeding areas"""
        min_distance = float('inf')
        
        for zone_lat, zone_lon, radius in self.high_risk_zones:
            # Simple distance calculation
            distance = np.sqrt((lat - zone_lat)**2 + (lon - zone_lon)**2)
            min_distance = min(min_distance, distance)
        
        # Risk decreases with distance
        if min_distance < 2.0:
            return 1.0  # Very high risk
        elif min_distance < 5.0:
            return 0.8  # High risk
        elif min_distance < 10.0:
            return 0.5  # Medium risk
        elif min_distance < 20.0:
            return 0.3  # Low-medium risk
        else:
            return 0.1  # Low risk
    
    def _calculate_temperature_risk(self, temp: float) -> float:
        """Optimal breeding temperature: 25-35°C"""
        if 25 <= temp <= 35:
            return 1.0  # Optimal
        elif 20 <= temp < 25 or 35 < temp <= 40:
            return 0.6  # Suboptimal but possible
        else:
            return 0.2  # Too cold or too hot
    
    def _calculate_rainfall_risk(self, rainfall: float) -> float:
        """10-30mm in 2 weeks triggers breeding"""
        if 10 <= rainfall <= 30:
            return 1.0  # Perfect trigger
        elif 5 <= rainfall < 10 or 30 < rainfall <= 50:
            return 0.6  # Possible
        elif 0 < rainfall < 5 or 50 < rainfall <= 100:
            return 0.3  # Less likely
        else:
            return 0.1  # Too dry or too wet
    
    def _calculate_vegetation_risk(self, ndvi: float) -> float:
        """NDVI > 0.4 indicates sufficient vegetation"""
        if ndvi >= 0.5:
            return 1.0  # Abundant food
        elif ndvi >= 0.4:
            return 0.8  # Good food availability
        elif ndvi >= 0.3:
            return 0.5  # Moderate
        elif ndvi >= 0.2:
            return 0.2  # Sparse
        else:
            return 0.05  # Very sparse
    
    def _calculate_humidity_risk(self, humidity: float) -> float:
        """>60% humidity helps egg survival"""
        if humidity >= 70:
            return 1.0  # Optimal
        elif humidity >= 60:
            return 0.8  # Good
        elif humidity >= 50:
            return 0.5  # Moderate
        else:
            return 0.2  # Too dry
    
    def _calculate_wind_risk(self, wind_speed: float) -> float:
        """Wind helps swarm migration (10-30 km/h optimal)"""
        if wind_speed is None:
            return 0.5
        
        if 10 <= wind_speed <= 30:
            return 0.9  # Optimal for migration
        elif 5 <= wind_speed < 10 or 30 < wind_speed <= 40:
            return 0.6  # Possible
        else:
            return 0.3  # Too calm or too strong
    
    def _categorize_risk(self, risk: float) -> Tuple[str, str]:
        """Categorize risk level"""
        if risk >= 0.75:
            return "CRITICAL", "🔴"
        elif risk >= 0.60:
            return "HIGH", "🟠"
        elif risk >= 0.40:
            return "MODERATE", "🟡"
        elif risk >= 0.20:
            return "LOW", "🟢"
        else:
            return "VERY LOW", "⚪"
    
    def _explain_factors(
        self, geo, temp, rain, veg, humid, wind, season
    ) -> List[Dict]:
        """Explain which factors contribute to risk"""
        factors = []
        
        if geo > 0.7:
            factors.append({
                'factor': 'Geographic Location',
                'status': 'high_risk',
                'message': 'Area is in known locust breeding zone'
            })
        
        if temp > 0.8:
            factors.append({
                'factor': 'Temperature',
                'status': 'favorable',
                'message': 'Temperature is optimal for locust breeding (25-35°C)'
            })
        
        if rain > 0.8:
            factors.append({
                'factor': 'Recent Rainfall',
                'status': 'favorable',
                'message': 'Recent rains have created breeding conditions'
            })
        
        if veg > 0.7:
            factors.append({
                'factor': 'Vegetation',
                'status': 'favorable',
                'message': 'Abundant vegetation provides food for locusts'
            })
        
        if humid > 0.7:
            factors.append({
                'factor': 'Humidity',
                'status': 'favorable',
                'message': 'High humidity supports egg survival'
            })
        
        if season > 0.8:
            factors.append({
                'factor': 'Season',
                'status': 'high_risk',
                'message': 'Peak locust season (breeding period)'
            })
        
        return factors
    
    def _generate_recommendations(
        self, risk: float, factors: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk >= 0.75:
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "Conduct daily field inspections for locust presence",
                "Notify local agriculture department immediately",
                "Prepare bio-pesticides/chemical sprays",
                "Coordinate with neighboring farmers",
                "Set up early warning surveillance system",
            ])
        elif risk >= 0.60:
            recommendations.extend([
                "⚠️ HIGH ALERT - Monitor closely",
                "Inspect fields every 2-3 days",
                "Report any locust sightings to authorities",
                "Prepare control equipment and supplies",
                "Join local farmer WhatsApp groups for updates",
            ])
        elif risk >= 0.40:
            recommendations.extend([
                "🟡 MODERATE RISK - Stay vigilant",
                "Weekly field inspections recommended",
                "Keep contact details of pest control services",
                "Monitor government locust alerts",
            ])
        else:
            recommendations.extend([
                "🟢 LOW RISK - Routine monitoring sufficient",
                "Monthly inspections adequate",
                "Stay informed through news/alerts",
            ])
        
        return recommendations
    
    def _calculate_inspection_interval(self, risk: float) -> int:
        """Recommend inspection interval in days"""
        if risk >= 0.75:
            return 1  # Daily
        elif risk >= 0.60:
            return 2  # Every 2 days
        elif risk >= 0.40:
            return 7  # Weekly
        elif risk >= 0.20:
            return 14  # Bi-weekly
        else:
            return 30  # Monthly
