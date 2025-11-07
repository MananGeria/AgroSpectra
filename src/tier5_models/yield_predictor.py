"""
Crop Yield Prediction Model
Predicts expected yield based on vegetation indices, weather, and historical data
"""

import numpy as np
from typing import Dict, List
from datetime import datetime


class YieldPredictor:
    """
    Predicts crop yield based on multiple factors
    Uses empirical models and machine learning approaches
    """
    
    def __init__(self):
        # Baseline yields (kg/hectare) for different crops under optimal conditions
        self.baseline_yields = {
            'wheat': 4500,
            'rice': 5500,
            'corn': 9000,
            'potato': 25000,
            'soybean': 3000,
            'cotton': 1800,
            'sugarcane': 70000,
            'barley': 4000,
            'millet': 1500,
            'sorghum': 2500
        }
        
        # Optimal NDVI ranges for maximum yield
        self.optimal_ndvi = {
            'wheat': 0.75,
            'rice': 0.80,
            'corn': 0.85,
            'potato': 0.70,
            'soybean': 0.72,
            'cotton': 0.68,
            'sugarcane': 0.82,
            'barley': 0.73,
            'millet': 0.65,
            'sorghum': 0.67
        }
    
    def predict_yield(
        self,
        crop_type: str,
        ndvi_mean: float,
        evi_mean: float,
        area_hectares: float,
        temperature_mean: float,
        rainfall_sum: float,
        growth_stage: str = 'vegetative',
        soil_quality: float = 0.7,  # 0-1 scale
        irrigation_available: bool = True,
        days_to_harvest: int = 60
    ) -> Dict:
        """
        Predict crop yield
        
        Args:
            crop_type: Type of crop
            ndvi_mean: Mean NDVI value
            evi_mean: Mean EVI value
            area_hectares: Field area in hectares
            temperature_mean: Average temperature (°C)
            rainfall_sum: Total rainfall (mm)
            growth_stage: Current growth stage
            soil_quality: Soil quality score (0-1)
            irrigation_available: Whether irrigation is available
            days_to_harvest: Estimated days remaining to harvest
            
        Returns:
            Dictionary with yield predictions and analysis
        """
        crop_type = crop_type.lower()
        
        if crop_type not in self.baseline_yields:
            return {
                'predicted_yield_kg_per_ha': 0,
                'total_yield_kg': 0,
                'confidence': 0.0,
                'message': f'Crop type {crop_type} not supported'
            }
        
        # Get baseline yield
        baseline = self.baseline_yields[crop_type]
        optimal_ndvi = self.optimal_ndvi[crop_type]
        
        # Calculate yield reduction/enhancement factors
        
        # 1. Vegetation Health Factor (based on NDVI)
        ndvi_factor = self._calculate_ndvi_factor(ndvi_mean, optimal_ndvi)
        
        # 2. Temperature Stress Factor
        temp_factor = self._calculate_temperature_factor(temperature_mean, crop_type)
        
        # 3. Water Availability Factor
        water_factor = self._calculate_water_factor(
            rainfall_sum, irrigation_available, crop_type
        )
        
        # 4. Soil Quality Factor
        soil_factor = 0.7 + (soil_quality * 0.3)  # Scale 0.7-1.0
        
        # 5. Growth Stage Factor
        stage_factor = self._get_growth_stage_factor(growth_stage, days_to_harvest)
        
        # Combined yield factor
        yield_factor = (
            ndvi_factor * 0.35 +
            temp_factor * 0.25 +
            water_factor * 0.25 +
            soil_factor * 0.10 +
            stage_factor * 0.05
        )
        
        # Predicted yield per hectare
        predicted_yield_per_ha = baseline * yield_factor
        
        # Total yield
        total_yield = predicted_yield_per_ha * area_hectares
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            ndvi_mean, temperature_mean, rainfall_sum, days_to_harvest
        )
        
        # Yield grade
        yield_grade = self._grade_yield(yield_factor)
        
        # Risk factors
        risk_factors = self._identify_risk_factors(
            ndvi_factor, temp_factor, water_factor, soil_factor
        )
        
        return {
            'predicted_yield_kg_per_ha': round(predicted_yield_per_ha, 2),
            'total_yield_kg': round(total_yield, 2),
            'total_yield_tons': round(total_yield / 1000, 2),
            'baseline_yield_kg_per_ha': baseline,
            'yield_factor': round(yield_factor, 3),
            'yield_grade': yield_grade,
            'confidence': round(confidence, 2),
            'contributing_factors': {
                'vegetation_health': round(ndvi_factor, 3),
                'temperature_stress': round(temp_factor, 3),
                'water_availability': round(water_factor, 3),
                'soil_quality': round(soil_factor, 3),
                'growth_stage': round(stage_factor, 3)
            },
            'risk_factors': risk_factors,
            'improvement_potential': self._suggest_improvements(
                ndvi_factor, temp_factor, water_factor, soil_factor
            ),
            'estimated_harvest_date': self._estimate_harvest_date(days_to_harvest),
            'economic_estimate': self._calculate_economic_value(
                total_yield, crop_type
            )
        }
    
    def _calculate_ndvi_factor(self, ndvi: float, optimal: float) -> float:
        """Calculate yield factor based on NDVI"""
        if ndvi >= optimal:
            return 1.0
        elif ndvi >= optimal * 0.8:
            return 0.85 + (ndvi - optimal * 0.8) / (optimal * 0.2) * 0.15
        elif ndvi >= optimal * 0.6:
            return 0.65 + (ndvi - optimal * 0.6) / (optimal * 0.2) * 0.20
        elif ndvi >= optimal * 0.4:
            return 0.40 + (ndvi - optimal * 0.4) / (optimal * 0.2) * 0.25
        else:
            return max(0.2, ndvi / (optimal * 0.4) * 0.40)
    
    def _calculate_temperature_factor(self, temp: float, crop_type: str) -> float:
        """Calculate yield impact from temperature"""
        # Optimal temperature ranges for different crops
        optimal_ranges = {
            'wheat': (15, 25),
            'rice': (25, 35),
            'corn': (20, 30),
            'potato': (15, 20),
            'soybean': (20, 30),
            'cotton': (25, 35),
            'sugarcane': (25, 35),
            'barley': (12, 22),
            'millet': (25, 35),
            'sorghum': (25, 35)
        }
        
        if crop_type not in optimal_ranges:
            return 0.8
        
        min_temp, max_temp = optimal_ranges[crop_type]
        
        if min_temp <= temp <= max_temp:
            return 1.0
        elif temp < min_temp:
            # Cold stress
            deviation = min_temp - temp
            return max(0.3, 1.0 - deviation * 0.05)
        else:
            # Heat stress
            deviation = temp - max_temp
            return max(0.3, 1.0 - deviation * 0.04)
    
    def _calculate_water_factor(
        self,
        rainfall: float,
        irrigation: bool,
        crop_type: str
    ) -> float:
        """Calculate yield impact from water availability"""
        # Water requirements (mm) for different crops per season
        water_needs = {
            'wheat': 450,
            'rice': 1200,
            'corn': 600,
            'potato': 500,
            'soybean': 500,
            'cotton': 700,
            'sugarcane': 1500,
            'barley': 450,
            'millet': 400,
            'sorghum': 450
        }
        
        need = water_needs.get(crop_type, 500)
        
        if irrigation:
            # With irrigation, can meet most water needs
            water_available = rainfall + (need * 0.6)  # Assume 60% irrigation supplement
        else:
            water_available = rainfall
        
        if water_available >= need:
            return 1.0
        else:
            water_ratio = water_available / need
            if water_ratio >= 0.8:
                return 0.85 + (water_ratio - 0.8) * 0.75
            elif water_ratio >= 0.6:
                return 0.65 + (water_ratio - 0.6) * 1.0
            else:
                return max(0.3, water_ratio * 1.08)
    
    def _get_growth_stage_factor(self, stage: str, days_to_harvest: int) -> float:
        """Adjust predictions based on growth stage"""
        stage_factors = {
            'germination': 0.70,  # Early stage, high uncertainty
            'vegetative': 0.85,
            'flowering': 0.95,
            'grain_filling': 1.00,  # Most predictable
            'maturity': 1.00
        }
        
        base_factor = stage_factors.get(stage, 0.85)
        
        # Reduce factor if harvest is far away (more uncertainty)
        if days_to_harvest > 90:
            base_factor *= 0.85
        elif days_to_harvest > 60:
            base_factor *= 0.92
        
        return base_factor
    
    def _calculate_confidence(
        self,
        ndvi: float,
        temp: float,
        rainfall: float,
        days_to_harvest: int
    ) -> float:
        """Calculate prediction confidence"""
        confidence = 0.9
        
        # Reduce confidence for extreme values
        if ndvi < 0.3 or ndvi > 0.95:
            confidence -= 0.15
        
        if temp < 10 or temp > 40:
            confidence -= 0.10
        
        # More uncertainty when harvest is far
        if days_to_harvest > 90:
            confidence -= 0.20
        elif days_to_harvest > 60:
            confidence -= 0.10
        
        return max(0.4, confidence)
    
    def _grade_yield(self, yield_factor: float) -> str:
        """Grade the predicted yield"""
        if yield_factor >= 0.90:
            return 'Excellent (>90% potential)'
        elif yield_factor >= 0.75:
            return 'Good (75-90% potential)'
        elif yield_factor >= 0.60:
            return 'Fair (60-75% potential)'
        elif yield_factor >= 0.45:
            return 'Below Average (45-60% potential)'
        else:
            return 'Poor (<45% potential)'
    
    def _identify_risk_factors(
        self,
        ndvi_factor: float,
        temp_factor: float,
        water_factor: float,
        soil_factor: float
    ) -> List[str]:
        """Identify factors limiting yield"""
        risks = []
        
        if ndvi_factor < 0.7:
            risks.append('⚠️ Poor vegetation health - Check for diseases/pests')
        
        if temp_factor < 0.7:
            risks.append('🌡️ Temperature stress - Consider heat/cold protection')
        
        if water_factor < 0.7:
            risks.append('💧 Water stress - Increase irrigation or improve water management')
        
        if soil_factor < 0.8:
            risks.append('🌱 Poor soil quality - Consider soil amendment and fertilization')
        
        if not risks:
            risks.append('✅ No major limiting factors detected')
        
        return risks
    
    def _suggest_improvements(
        self,
        ndvi_factor: float,
        temp_factor: float,
        water_factor: float,
        soil_factor: float
    ) -> List[str]:
        """Suggest ways to improve yield"""
        suggestions = []
        
        if ndvi_factor < 0.85:
            suggestions.append('Apply balanced fertilizer to improve plant health')
            suggestions.append('Implement integrated pest and disease management')
        
        if water_factor < 0.85:
            suggestions.append('Optimize irrigation schedule')
            suggestions.append('Consider mulching to conserve soil moisture')
        
        if soil_factor < 0.9:
            suggestions.append('Conduct soil testing and apply amendments')
            suggestions.append('Add organic matter to improve soil health')
        
        if temp_factor < 0.85:
            suggestions.append('Use shade nets or protective covers during extreme weather')
        
        suggestions.append('Monitor crop regularly and take timely corrective actions')
        
        return suggestions
    
    def _estimate_harvest_date(self, days_to_harvest: int) -> str:
        """Estimate harvest date"""
        from datetime import datetime, timedelta
        harvest_date = datetime.now() + timedelta(days=days_to_harvest)
        return harvest_date.strftime('%Y-%m-%d')
    
    def _calculate_economic_value(self, total_yield_kg: float, crop_type: str) -> Dict:
        """Estimate economic value of predicted yield"""
        # Average market prices (₹/kg or $/kg - can be adjusted)
        market_prices = {
            'wheat': 25,  # ₹/kg
            'rice': 30,
            'corn': 20,
            'potato': 15,
            'soybean': 40,
            'cotton': 60,
            'sugarcane': 3,
            'barley': 22,
            'millet': 25,
            'sorghum': 20
        }
        
        price_per_kg = market_prices.get(crop_type, 25)
        gross_value = total_yield_kg * price_per_kg
        
        return {
            'estimated_gross_value': round(gross_value, 2),
            'price_per_kg': price_per_kg,
            'currency': 'INR',  # Change as needed
            'note': 'Based on average market prices - actual prices may vary'
        }
