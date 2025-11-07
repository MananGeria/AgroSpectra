"""
Tier 5: AI Modeling Layer - Fusion Engine
Combines CNN and LSTM predictions into unified Crop Health Score
"""

import numpy as np
from typing import Dict, Tuple
import yaml
from loguru import logger


class FusionEngine:
    """Fusion engine for combining CNN and LSTM predictions"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['models']['fusion_engine']
        
        self.weights = self.config['weights']
        self.thresholds = self.config['thresholds']
    
    def compute_crop_health_score(
        self,
        cnn_probabilities: Dict[str, float],
        lstm_pest_probability: float
    ) -> Dict[str, float]:
        """
        Compute unified Crop Health Score
        
        Formula: CHS = 0.6 * P_CNN(Healthy) + 0.4 * (1 - P_LSTM(Pest))
        
        Args:
            cnn_probabilities: Dictionary with CNN predicted probabilities
                              {'healthy': p1, 'stressed': p2, 'diseased': p3}
            lstm_pest_probability: LSTM predicted pest outbreak probability
            
        Returns:
            Dictionary with:
                - crop_health_score: Overall score (0-1)
                - health_component: Contribution from crop health
                - pest_component: Contribution from pest risk
                - risk_level: Categorized risk level
                - recommendation: Recommended action
        """
        # Extract healthy probability from CNN
        p_healthy = cnn_probabilities.get('healthy', 0.0)
        
        # Compute components
        health_component = self.weights['cnn_healthy'] * p_healthy
        pest_component = self.weights['lstm_pest'] * (1 - lstm_pest_probability)
        
        # Compute overall score
        crop_health_score = health_component + pest_component
        
        # Determine risk level
        risk_level = self._categorize_risk(crop_health_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            crop_health_score,
            cnn_probabilities,
            lstm_pest_probability
        )
        
        result = {
            'crop_health_score': float(crop_health_score),
            'health_component': float(health_component),
            'pest_component': float(pest_component),
            'cnn_prediction': cnn_probabilities.get('predicted_class', 'unknown'),
            'cnn_confidence': cnn_probabilities.get('confidence', 0.0),
            'pest_risk_probability': float(lstm_pest_probability),
            'risk_level': risk_level,
            'recommendation': recommendation
        }
        
        logger.debug(f"Fusion result: CHS={crop_health_score:.3f}, Risk={risk_level}")
        
        return result
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize crop health score into risk levels"""
        thresholds = self.thresholds
        
        if score >= thresholds['excellent']:
            return 'excellent'
        elif score >= thresholds['good']:
            return 'good'
        elif score >= thresholds['fair']:
            return 'fair'
        elif score >= thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_recommendation(
        self,
        score: float,
        cnn_probs: Dict[str, float],
        pest_prob: float
    ) -> str:
        """Generate actionable recommendation based on predictions"""
        recommendations = []
        
        # Check crop health status
        if cnn_probs.get('diseased', 0) > 0.5:
            recommendations.append(
                "⚠️ Disease detected. Inspect crops and apply appropriate treatment."
            )
        elif cnn_probs.get('stressed', 0) > 0.5:
            recommendations.append(
                "⚡ Crop stress detected. Check irrigation, nutrients, and environmental conditions."
            )
        
        # Check pest risk
        if pest_prob > 0.75:
            recommendations.append(
                "🐛 High pest outbreak risk! Monitor closely and prepare pest management interventions."
            )
        elif pest_prob > 0.5:
            recommendations.append(
                "🐛 Moderate pest risk. Increase monitoring frequency."
            )
        
        # General recommendation based on score
        if score >= 0.8:
            if not recommendations:
                recommendations.append(
                    "✅ Crops are in excellent condition. Continue current management practices."
                )
        elif score >= 0.6:
            if not recommendations:
                recommendations.append(
                    "✓ Crops are healthy. Maintain regular monitoring and care."
                )
        elif score >= 0.4:
            if not recommendations:
                recommendations.append(
                    "⚠️ Attention needed. Investigate potential issues affecting crop health."
                )
        else:
            if not recommendations:
                recommendations.append(
                    "❌ Critical condition. Immediate intervention required!"
                )
        
        return " | ".join(recommendations)
    
    def batch_fusion(
        self,
        cnn_predictions: list,
        lstm_predictions: list
    ) -> list:
        """
        Compute fusion for batch of predictions
        
        Args:
            cnn_predictions: List of CNN prediction dictionaries
            lstm_predictions: List of LSTM pest probabilities
            
        Returns:
            List of fusion results
        """
        if len(cnn_predictions) != len(lstm_predictions):
            raise ValueError("CNN and LSTM predictions must have same length")
        
        results = []
        for cnn_pred, lstm_pred in zip(cnn_predictions, lstm_predictions):
            result = self.compute_crop_health_score(cnn_pred, lstm_pred)
            results.append(result)
        
        return results
    
    def compute_spatial_statistics(self, fusion_results: list) -> Dict:
        """
        Compute spatial statistics from multiple fusion results
        
        Args:
            fusion_results: List of fusion result dictionaries
            
        Returns:
            Dictionary with spatial statistics
        """
        if not fusion_results:
            return {}
        
        scores = [r['crop_health_score'] for r in fusion_results]
        pest_probs = [r['pest_risk_probability'] for r in fusion_results]
        
        # Count risk levels
        risk_counts = {}
        for result in fusion_results:
            level = result['risk_level']
            risk_counts[level] = risk_counts.get(level, 0) + 1
        
        stats = {
            'mean_crop_health_score': float(np.mean(scores)),
            'std_crop_health_score': float(np.std(scores)),
            'min_crop_health_score': float(np.min(scores)),
            'max_crop_health_score': float(np.max(scores)),
            'mean_pest_risk': float(np.mean(pest_probs)),
            'max_pest_risk': float(np.max(pest_probs)),
            'risk_level_counts': risk_counts,
            'total_pixels': len(fusion_results),
            'healthy_percentage': (
                risk_counts.get('excellent', 0) + risk_counts.get('good', 0)
            ) / len(fusion_results) * 100,
            'critical_percentage': (
                risk_counts.get('poor', 0) + risk_counts.get('critical', 0)
            ) / len(fusion_results) * 100
        }
        
        return stats


class PredictionAggregator:
    """Aggregate predictions across time and space"""
    
    @staticmethod
    def temporal_aggregation(
        predictions: list,
        dates: list,
        method: str = 'mean'
    ) -> Dict:
        """
        Aggregate predictions over time
        
        Args:
            predictions: List of prediction dictionaries
            dates: Corresponding dates
            method: Aggregation method ('mean', 'median', 'max', 'min')
            
        Returns:
            Aggregated statistics
        """
        scores = [p['crop_health_score'] for p in predictions]
        pest_risks = [p['pest_risk_probability'] for p in predictions]
        
        if method == 'mean':
            agg_score = np.mean(scores)
            agg_pest = np.mean(pest_risks)
        elif method == 'median':
            agg_score = np.median(scores)
            agg_pest = np.median(pest_risks)
        elif method == 'max':
            agg_score = np.max(scores)
            agg_pest = np.max(pest_risks)
        elif method == 'min':
            agg_score = np.min(scores)
            agg_pest = np.min(pest_risks)
        else:
            agg_score = np.mean(scores)
            agg_pest = np.mean(pest_risks)
        
        return {
            'aggregated_crop_health_score': float(agg_score),
            'aggregated_pest_risk': float(agg_pest),
            'temporal_trend': 'improving' if scores[-1] > scores[0] else 'declining',
            'date_range': (min(dates), max(dates)),
            'n_observations': len(predictions)
        }
    
    @staticmethod
    def spatial_aggregation(
        predictions: list,
        coordinates: list,
        method: str = 'mean'
    ) -> Dict:
        """
        Aggregate predictions across space
        
        Args:
            predictions: List of prediction dictionaries
            coordinates: Corresponding spatial coordinates
            method: Aggregation method
            
        Returns:
            Spatially aggregated statistics
        """
        scores = [p['crop_health_score'] for p in predictions]
        
        if method == 'mean':
            agg_score = np.mean(scores)
        elif method == 'median':
            agg_score = np.median(scores)
        else:
            agg_score = np.mean(scores)
        
        # Identify hotspots (low scores)
        score_threshold = 0.4
        hotspot_indices = [i for i, s in enumerate(scores) if s < score_threshold]
        hotspot_coords = [coordinates[i] for i in hotspot_indices]
        
        return {
            'spatial_mean_score': float(agg_score),
            'spatial_variability': float(np.std(scores)),
            'n_hotspots': len(hotspot_indices),
            'hotspot_coordinates': hotspot_coords,
            'hotspot_percentage': len(hotspot_indices) / len(scores) * 100
        }


if __name__ == "__main__":
    # Example usage
    fusion = FusionEngine()
    
    # Example CNN prediction
    cnn_pred = {
        'healthy': 0.7,
        'stressed': 0.2,
        'diseased': 0.1,
        'predicted_class': 'healthy',
        'confidence': 0.7
    }
    
    # Example LSTM prediction
    lstm_pred = 0.3  # 30% pest risk
    
    # Compute fusion
    result = fusion.compute_crop_health_score(cnn_pred, lstm_pred)
    
    print("\nFusion Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
