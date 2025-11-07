"""
Multi-Crop Disease Detection Model
Detects various crop diseases from spectral signatures and visual symptoms
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class MultiCropDiseaseDetector:
    """
    Detects crop diseases across multiple crop types
    Uses spectral indices and pattern recognition
    """
    
    def __init__(self):
        # Disease database with spectral signatures
        self.disease_database = {
            'wheat': {
                'rust': {
                    'ndvi_range': (0.3, 0.5),
                    'red_edge_shift': -5,  # nm shift
                    'symptoms': ['yellow-orange pustules', 'leaf chlorosis'],
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7}
                },
                'blight': {
                    'ndvi_range': (0.2, 0.4),
                    'red_edge_shift': -8,
                    'symptoms': ['water-soaked lesions', 'leaf necrosis'],
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.5, 'severe': 0.75}
                },
                'powdery_mildew': {
                    'ndvi_range': (0.4, 0.6),
                    'red_edge_shift': -3,
                    'symptoms': ['white powdery coating', 'leaf distortion'],
                    'severity_thresholds': {'mild': 0.2, 'moderate': 0.4, 'severe': 0.6}
                }
            },
            'rice': {
                'blast': {
                    'ndvi_range': (0.25, 0.45),
                    'red_edge_shift': -7,
                    'symptoms': ['diamond-shaped lesions', 'leaf collapse'],
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.55, 'severe': 0.8}
                },
                'bacterial_blight': {
                    'ndvi_range': (0.2, 0.4),
                    'red_edge_shift': -10,
                    'symptoms': ['water-soaked stripes', 'leaf yellowing'],
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.5, 'severe': 0.75}
                },
                'sheath_blight': {
                    'ndvi_range': (0.3, 0.5),
                    'red_edge_shift': -6,
                    'symptoms': ['elliptical lesions on sheath', 'rotting'],
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7}
                }
            },
            'corn': {
                'northern_leaf_blight': {
                    'ndvi_range': (0.3, 0.5),
                    'red_edge_shift': -6,
                    'symptoms': ['long cigar-shaped lesions', 'leaf death'],
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.55, 'severe': 0.75}
                },
                'gray_leaf_spot': {
                    'ndvi_range': (0.25, 0.45),
                    'red_edge_shift': -7,
                    'symptoms': ['rectangular gray lesions', 'premature death'],
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.5, 'severe': 0.7}
                },
                'common_rust': {
                    'ndvi_range': (0.35, 0.55),
                    'red_edge_shift': -4,
                    'symptoms': ['brown pustules', 'leaf chlorosis'],
                    'severity_thresholds': {'mild': 0.2, 'moderate': 0.4, 'severe': 0.6}
                }
            },
            'potato': {
                'late_blight': {
                    'ndvi_range': (0.2, 0.4),
                    'red_edge_shift': -9,
                    'symptoms': ['water-soaked lesions', 'white fungal growth'],
                    'severity_thresholds': {'mild': 0.35, 'moderate': 0.6, 'severe': 0.85}
                },
                'early_blight': {
                    'ndvi_range': (0.3, 0.5),
                    'red_edge_shift': -5,
                    'symptoms': ['concentric ring lesions', 'target-like spots'],
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.5, 'severe': 0.75}
                }
            }
        }
    
    def detect_diseases(
        self,
        crop_type: str,
        ndvi: float,
        evi: float,
        red_edge_position: float = 720,  # Default red edge position (nm)
        temperature: float = 25.0,
        humidity: float = 65.0,
        month: int = 6
    ) -> Dict:
        """
        Detect diseases for a specific crop
        
        Args:
            crop_type: Type of crop (wheat, rice, corn, potato)
            ndvi: Normalized Difference Vegetation Index
            evi: Enhanced Vegetation Index
            red_edge_position: Red edge spectral position (nm)
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage
            month: Current month (1-12)
            
        Returns:
            Dictionary with detected diseases and confidence scores
        """
        crop_type = crop_type.lower()
        
        if crop_type not in self.disease_database:
            return {
                'detected_diseases': [],
                'overall_health': 'unknown',
                'confidence': 0.0,
                'message': f'Crop type {crop_type} not in database'
            }
        
        detected_diseases = []
        crop_diseases = self.disease_database[crop_type]
        
        # Check each disease for the crop
        for disease_name, disease_info in crop_diseases.items():
            # Check if NDVI matches disease signature
            ndvi_min, ndvi_max = disease_info['ndvi_range']
            
            if ndvi_min <= ndvi <= ndvi_max:
                # Calculate confidence based on multiple factors
                confidence = self._calculate_disease_confidence(
                    ndvi, evi, red_edge_position,
                    disease_info, temperature, humidity, month
                )
                
                if confidence > 0.3:  # Threshold for detection
                    severity = self._assess_severity(confidence, disease_info['severity_thresholds'])
                    
                    detected_diseases.append({
                        'disease': disease_name.replace('_', ' ').title(),
                        'confidence': round(confidence, 2),
                        'severity': severity,
                        'symptoms': disease_info['symptoms'],
                        'recommended_actions': self._get_treatment_recommendations(
                            disease_name, severity, crop_type
                        )
                    })
        
        # Sort by confidence
        detected_diseases.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Determine overall health
        if not detected_diseases:
            overall_health = 'healthy'
        elif max(d['confidence'] for d in detected_diseases) > 0.7:
            overall_health = 'diseased'
        elif max(d['confidence'] for d in detected_diseases) > 0.5:
            overall_health = 'at_risk'
        else:
            overall_health = 'monitoring_required'
        
        return {
            'detected_diseases': detected_diseases,
            'overall_health': overall_health,
            'total_detections': len(detected_diseases),
            'highest_confidence': detected_diseases[0]['confidence'] if detected_diseases else 0.0,
            'environmental_factors': {
                'temperature': temperature,
                'humidity': humidity,
                'favorable_for_disease': self._check_disease_favorable_conditions(
                    temperature, humidity, month
                )
            }
        }
    
    def _calculate_disease_confidence(
        self,
        ndvi: float,
        evi: float,
        red_edge: float,
        disease_info: Dict,
        temperature: float,
        humidity: float,
        month: int
    ) -> float:
        """Calculate confidence score for disease detection"""
        
        # NDVI match score (0-1)
        ndvi_min, ndvi_max = disease_info['ndvi_range']
        ndvi_center = (ndvi_min + ndvi_max) / 2
        ndvi_score = 1.0 - min(abs(ndvi - ndvi_center) / (ndvi_max - ndvi_min), 1.0)
        
        # Red edge shift score
        expected_shift = disease_info['red_edge_shift']
        actual_shift = red_edge - 720  # 720 is baseline red edge
        shift_match = 1.0 - min(abs(actual_shift - expected_shift) / 10.0, 1.0)
        
        # Environmental favorability (diseases thrive in certain conditions)
        env_score = 0.0
        if 20 <= temperature <= 30 and humidity > 70:
            env_score = 0.8  # Favorable for most fungal diseases
        elif 15 <= temperature <= 25 and humidity > 60:
            env_score = 0.5
        else:
            env_score = 0.2
        
        # Seasonal factor (some diseases are seasonal)
        seasonal_score = 1.0
        if month in [6, 7, 8, 9]:  # Monsoon/humid season
            seasonal_score = 1.2
        elif month in [12, 1, 2]:  # Winter
            seasonal_score = 0.7
        
        # Combined confidence
        confidence = (
            ndvi_score * 0.4 +
            shift_match * 0.3 +
            env_score * 0.2 +
            (seasonal_score - 0.7) * 0.1
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _assess_severity(self, confidence: float, thresholds: Dict) -> str:
        """Assess disease severity based on confidence and thresholds"""
        if confidence >= thresholds['severe']:
            return 'severe'
        elif confidence >= thresholds['moderate']:
            return 'moderate'
        elif confidence >= thresholds['mild']:
            return 'mild'
        else:
            return 'trace'
    
    def _get_treatment_recommendations(
        self,
        disease_name: str,
        severity: str,
        crop_type: str
    ) -> List[str]:
        """Get treatment recommendations for detected disease"""
        
        recommendations = []
        
        if severity in ['severe', 'moderate']:
            recommendations.append("🔴 Immediate action required")
            recommendations.append("Apply appropriate fungicide/bactericide")
            recommendations.append("Remove and destroy severely infected plants")
            recommendations.append("Improve field drainage if waterlogging present")
        else:
            recommendations.append("🟡 Monitor closely for spread")
            recommendations.append("Consider preventive fungicide application")
            recommendations.append("Maintain optimal spacing for air circulation")
        
        # Disease-specific recommendations
        if 'rust' in disease_name:
            recommendations.append("Use rust-resistant varieties in next season")
            recommendations.append("Apply sulfur-based or triazole fungicides")
        elif 'blight' in disease_name:
            recommendations.append("Ensure proper crop rotation")
            recommendations.append("Avoid overhead irrigation")
            recommendations.append("Apply copper-based or systemic fungicides")
        elif 'mildew' in disease_name:
            recommendations.append("Reduce nitrogen fertilization")
            recommendations.append("Improve air circulation around plants")
        
        return recommendations
    
    def _check_disease_favorable_conditions(
        self,
        temperature: float,
        humidity: float,
        month: int
    ) -> bool:
        """Check if environmental conditions favor disease development"""
        
        # Most fungal diseases thrive in warm, humid conditions
        if temperature >= 20 and temperature <= 30 and humidity > 70:
            return True
        
        # Monsoon season (high disease pressure)
        if month in [6, 7, 8, 9] and humidity > 65:
            return True
        
        return False
    
    def get_preventive_measures(self, crop_type: str) -> List[str]:
        """Get preventive measures for crop diseases"""
        
        general_measures = [
            "✅ Use certified disease-free seeds",
            "✅ Implement proper crop rotation",
            "✅ Maintain optimal plant spacing",
            "✅ Ensure good field drainage",
            "✅ Remove crop residues after harvest",
            "✅ Monitor fields regularly for early detection",
            "✅ Use resistant/tolerant varieties when available",
            "✅ Avoid excessive nitrogen fertilization",
            "✅ Implement integrated pest management (IPM)"
        ]
        
        return general_measures
