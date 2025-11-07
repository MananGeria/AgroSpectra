"""
Nutrient Deficiency Detection
Detects crop nutrient deficiencies from spectral signatures
"""

import numpy as np
from typing import Dict, List, Tuple


class NutrientDeficiencyDetector:
    """
    Detects nutrient deficiencies in crops using spectral analysis
    """
    
    def __init__(self):
        # Spectral signatures for different nutrient deficiencies
        self.deficiency_signatures = {
            'nitrogen': {
                'ndvi_threshold': 0.55,  # Below this suggests N deficiency
                'red_reflectance': 'high',  # Increased red reflectance
                'nir_reflectance': 'low',  # Decreased NIR
                'visual_symptoms': [
                    'Yellowing of older leaves (chlorosis)',
                    'Stunted growth',
                    'Light green to yellow plant color',
                    'Reduced tillering'
                ],
                'severity_impact': 'high',
                'mobility': 'mobile'  # Moves from old to new tissue
            },
            'phosphorus': {
                'ndvi_threshold': 0.60,
                'red_reflectance': 'moderate',
                'nir_reflectance': 'moderate_low',
                'visual_symptoms': [
                    'Dark green or purplish leaves',
                    'Stunted growth and delayed maturity',
                    'Poor root development',
                    'Leaf tips appear scorched'
                ],
                'severity_impact': 'high',
                'mobility': 'mobile'
            },
            'potassium': {
                'ndvi_threshold': 0.62,
                'red_reflectance': 'moderate_high',
                'nir_reflectance': 'moderate',
                'visual_symptoms': [
                    'Yellowing/browning of leaf margins',
                    'Scorched appearance on edges',
                    'Weak stems and lodging',
                    'Increased disease susceptibility'
                ],
                'severity_impact': 'medium',
                'mobility': 'mobile'
            },
            'magnesium': {
                'ndvi_threshold': 0.65,
                'red_reflectance': 'moderate',
                'nir_reflectance': 'moderate',
                'visual_symptoms': [
                    'Interveinal chlorosis (veins stay green)',
                    'Older leaves affected first',
                    'Yellowing between leaf veins',
                    'Upward leaf curling'
                ],
                'severity_impact': 'medium',
                'mobility': 'mobile'
            },
            'sulfur': {
                'ndvi_threshold': 0.63,
                'red_reflectance': 'moderate_high',
                'nir_reflectance': 'moderate_low',
                'visual_symptoms': [
                    'Uniform yellowing of younger leaves',
                    'Similar to N deficiency but affects new growth',
                    'Thin, spindly stems',
                    'Delayed maturity'
                ],
                'severity_impact': 'medium',
                'mobility': 'immobile'
            },
            'iron': {
                'ndvi_threshold': 0.68,
                'red_reflectance': 'low',
                'nir_reflectance': 'moderate',
                'visual_symptoms': [
                    'Interveinal chlorosis in young leaves',
                    'Veins remain green',
                    'White or yellow new leaves',
                    'Stunted growth in severe cases'
                ],
                'severity_impact': 'low',
                'mobility': 'immobile'
            },
            'zinc': {
                'ndvi_threshold': 0.67,
                'red_reflectance': 'moderate',
                'nir_reflectance': 'moderate',
                'visual_symptoms': [
                    'Short internodes (rosetting)',
                    'Small, distorted leaves',
                    'Interveinal chlorosis',
                    'Bronzing of leaves'
                ],
                'severity_impact': 'medium',
                'mobility': 'immobile'
            },
            'manganese': {
                'ndvi_threshold': 0.66,
                'red_reflectance': 'moderate',
                'nir_reflectance': 'moderate',
                'visual_symptoms': [
                    'Interveinal chlorosis in young leaves',
                    'Gray or tan necrotic spots',
                    'Poor seed development',
                    'Increased disease susceptibility'
                ],
                'severity_impact': 'low',
                'mobility': 'immobile'
            }
        }
    
    def detect_deficiencies(
        self,
        ndvi: float,
        evi: float,
        red_band: float = 0.1,  # Normalized reflectance (0-1)
        nir_band: float = 0.5,
        crop_type: str = 'general',
        soil_ph: float = 6.5,
        temperature: float = 25.0
    ) -> Dict:
        """
        Detect nutrient deficiencies
        
        Args:
            ndvi: Normalized Difference Vegetation Index
            evi: Enhanced Vegetation Index
            red_band: Red band reflectance (normalized)
            nir_band: Near-infrared band reflectance
            crop_type: Type of crop being analyzed
            soil_ph: Soil pH (affects nutrient availability)
            temperature: Temperature in Celsius
            
        Returns:
            Dictionary with detected deficiencies and recommendations
        """
        detected_deficiencies = []
        
        # Calculate derived indices
        red_edge_ndvi = (nir_band - red_band) / (nir_band + red_band)
        
        # Check each nutrient
        for nutrient, signature in self.deficiency_signatures.items():
            # Primary check: NDVI threshold
            if ndvi < signature['ndvi_threshold']:
                # Calculate deficiency probability
                probability = self._calculate_deficiency_probability(
                    nutrient, ndvi, red_band, nir_band, soil_ph, temperature
                )
                
                if probability > 0.3:  # Detection threshold
                    severity = self._assess_deficiency_severity(probability)
                    
                    detected_deficiencies.append({
                        'nutrient': nutrient.title(),
                        'probability': round(probability, 2),
                        'severity': severity,
                        'visual_symptoms': signature['visual_symptoms'],
                        'mobility': signature['mobility'],
                        'treatment': self._get_treatment_recommendation(
                            nutrient, severity, crop_type, soil_ph
                        )
                    })
        
        # Sort by probability
        detected_deficiencies.sort(key=lambda x: x['probability'], reverse=True)
        
        # Overall nutritional health
        if not detected_deficiencies:
            health_status = 'optimal'
        elif max(d['probability'] for d in detected_deficiencies) > 0.7:
            health_status = 'deficient'
        elif max(d['probability'] for d in detected_deficiencies) > 0.5:
            health_status = 'marginal'
        else:
            health_status = 'adequate'
        
        return {
            'detected_deficiencies': detected_deficiencies,
            'nutritional_health': health_status,
            'total_deficiencies': len(detected_deficiencies),
            'soil_factors': {
                'ph': soil_ph,
                'ph_impact': self._assess_ph_impact(soil_ph),
                'temperature': temperature
            },
            'recommended_actions': self._get_general_recommendations(
                detected_deficiencies, soil_ph
            )
        }
    
    def _calculate_deficiency_probability(
        self,
        nutrient: str,
        ndvi: float,
        red: float,
        nir: float,
        ph: float,
        temp: float
    ) -> float:
        """Calculate probability of specific nutrient deficiency"""
        
        signature = self.deficiency_signatures[nutrient]
        threshold = signature['ndvi_threshold']
        
        # Base probability from NDVI deviation
        ndvi_deviation = (threshold - ndvi) / threshold
        base_prob = min(ndvi_deviation * 2, 1.0)
        
        # Adjust for soil pH (affects nutrient availability)
        ph_factor = self._get_ph_availability_factor(nutrient, ph)
        
        # Temperature factor (affects nutrient uptake)
        temp_factor = 1.0
        if temp < 10 or temp > 35:
            temp_factor = 0.7  # Reduced uptake in extreme temps
        
        # Spectral signature matching
        spectral_factor = 1.0
        if nutrient == 'nitrogen':
            # N deficiency shows high red, low NIR
            if red > 0.12 and nir < 0.45:
                spectral_factor = 1.3
        elif nutrient == 'phosphorus':
            # P deficiency affects early growth
            if ndvi < 0.5:
                spectral_factor = 1.2
        
        # Combined probability
        probability = base_prob * ph_factor * temp_factor * spectral_factor * 0.8
        
        return np.clip(probability, 0.0, 1.0)
    
    def _get_ph_availability_factor(self, nutrient: str, ph: float) -> float:
        """
        Get nutrient availability factor based on soil pH
        Different nutrients have different pH optima
        """
        # Optimal pH ranges for nutrient availability
        ph_optima = {
            'nitrogen': (6.0, 7.5),
            'phosphorus': (6.5, 7.0),  # Most critical
            'potassium': (6.0, 7.5),
            'magnesium': (6.5, 7.5),
            'sulfur': (6.0, 8.0),
            'iron': (5.0, 6.5),  # Less available in alkaline soil
            'zinc': (5.5, 7.0),
            'manganese': (5.0, 6.5)
        }
        
        if nutrient not in ph_optima:
            return 1.0
        
        optimal_min, optimal_max = ph_optima[nutrient]
        
        if optimal_min <= ph <= optimal_max:
            return 0.8  # Good availability, lower deficiency probability
        elif ph < optimal_min:
            deviation = optimal_min - ph
            return 1.0 + min(deviation * 0.15, 0.4)
        else:
            deviation = ph - optimal_max
            # Some nutrients (Fe, Zn, Mn) become very unavailable in alkaline soil
            if nutrient in ['iron', 'zinc', 'manganese']:
                return 1.0 + min(deviation * 0.25, 0.6)
            else:
                return 1.0 + min(deviation * 0.1, 0.3)
    
    def _assess_deficiency_severity(self, probability: float) -> str:
        """Assess deficiency severity"""
        if probability >= 0.75:
            return 'severe'
        elif probability >= 0.55:
            return 'moderate'
        elif probability >= 0.35:
            return 'mild'
        else:
            return 'marginal'
    
    def _get_treatment_recommendation(
        self,
        nutrient: str,
        severity: str,
        crop_type: str,
        soil_ph: float
    ) -> Dict:
        """Get treatment recommendations for deficiency"""
        
        # Fertilizer recommendations
        fertilizers = {
            'nitrogen': {
                'quick_fix': 'Urea (46-0-0) or Ammonium Nitrate (34-0-0)',
                'application_rate': '50-150 kg/ha depending on severity',
                'method': 'Split application - apply in 2-3 doses',
                'timing': 'Apply at tillering and flowering stages'
            },
            'phosphorus': {
                'quick_fix': 'Single Super Phosphate (SSP 16% P2O5) or DAP (18-46-0)',
                'application_rate': '40-80 kg P2O5/ha',
                'method': 'Band placement near roots',
                'timing': 'Apply at sowing or planting'
            },
            'potassium': {
                'quick_fix': 'Muriate of Potash (MOP 60% K2O) or SOP',
                'application_rate': '30-60 kg K2O/ha',
                'method': 'Broadcast and incorporate',
                'timing': 'Apply at sowing and top-dress at vegetative stage'
            },
            'magnesium': {
                'quick_fix': 'Magnesium Sulfate (Epsom Salt) or Dolomitic Lime',
                'application_rate': '10-25 kg/ha',
                'method': 'Foliar spray (2% solution) or soil application',
                'timing': 'Immediate foliar application for quick results'
            },
            'sulfur': {
                'quick_fix': 'Gypsum (CaSO4) or Elemental Sulfur',
                'application_rate': '20-40 kg S/ha',
                'method': 'Broadcast and incorporate',
                'timing': 'Apply before sowing or as top-dress'
            },
            'iron': {
                'quick_fix': 'Iron Chelate (Fe-EDTA) or Ferrous Sulfate',
                'application_rate': '2-5 kg/ha (chelate) or 10-20 kg/ha (sulfate)',
                'method': 'Foliar spray (0.5% solution) preferred',
                'timing': 'Repeat application every 10-15 days until symptoms disappear'
            },
            'zinc': {
                'quick_fix': 'Zinc Sulfate (21% Zn) or Zinc Chelate',
                'application_rate': '10-25 kg ZnSO4/ha or 2-5 kg chelate/ha',
                'method': 'Soil application or foliar spray (0.5% solution)',
                'timing': 'Apply at sowing and as foliar spray at vegetative stage'
            },
            'manganese': {
                'quick_fix': 'Manganese Sulfate (26-28% Mn)',
                'application_rate': '5-10 kg/ha',
                'method': 'Foliar spray (0.3-0.5% solution)',
                'timing': 'Apply when symptoms appear, repeat if needed'
            }
        }
        
        treatment = fertilizers.get(nutrient, {})
        
        # Add urgency based on severity
        if severity == 'severe':
            urgency = '🔴 URGENT: Apply treatment immediately'
        elif severity == 'moderate':
            urgency = '🟠 Apply treatment within 1 week'
        else:
            urgency = '🟡 Monitor and apply treatment as needed'
        
        # pH correction recommendations
        ph_advice = []
        if soil_ph < 5.5:
            ph_advice.append('Soil is too acidic - apply lime to raise pH to 6.0-6.5')
        elif soil_ph > 7.5:
            ph_advice.append('Soil is too alkaline - apply elemental sulfur or acidic fertilizers')
        
        return {
            'urgency': urgency,
            'fertilizer': treatment.get('quick_fix', 'Consult agronomist'),
            'application_rate': treatment.get('application_rate', 'Follow package instructions'),
            'method': treatment.get('method', 'As per standard practice'),
            'timing': treatment.get('timing', 'As soon as possible'),
            'ph_correction': ph_advice
        }
    
    def _assess_ph_impact(self, ph: float) -> str:
        """Assess soil pH impact on nutrient availability"""
        if ph < 5.5:
            return 'Strongly Acidic - Limits availability of N, P, K, S, Ca, Mg'
        elif ph < 6.0:
            return 'Moderately Acidic - Optimal for micronutrients but may limit macronutrients'
        elif 6.0 <= ph <= 7.0:
            return 'Optimal - Best availability for most nutrients'
        elif ph <= 7.5:
            return 'Slightly Alkaline - Still acceptable for most crops'
        elif ph <= 8.0:
            return 'Moderately Alkaline - Limits Fe, Zn, Mn, Cu availability'
        else:
            return 'Strongly Alkaline - Severely limits micronutrient availability'
    
    def _get_general_recommendations(
        self,
        deficiencies: List[Dict],
        soil_ph: float
    ) -> List[str]:
        """Get general management recommendations"""
        
        recommendations = []
        
        if deficiencies:
            recommendations.append('📋 Conduct comprehensive soil testing')
            recommendations.append('💉 Apply recommended fertilizers based on severity')
            recommendations.append('🔄 Use balanced fertilization (N-P-K + micronutrients)')
        
        if soil_ph < 6.0:
            recommendations.append('🧪 Apply lime to correct soil acidity')
        elif soil_ph > 7.5:
            recommendations.append('🧪 Apply sulfur or acidifying amendments')
        
        recommendations.extend([
            '🌱 Apply organic matter to improve nutrient retention',
            '💧 Ensure adequate irrigation for nutrient uptake',
            '📅 Follow crop-specific fertilization schedule',
            '🔬 Conduct tissue testing to confirm deficiencies',
            '♻️ Implement crop rotation to maintain soil fertility'
        ])
        
        return recommendations
