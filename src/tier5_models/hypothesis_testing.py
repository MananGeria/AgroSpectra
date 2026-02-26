"""
Hypothesis Testing Module for AgroSpectra
Statistical validation of fusion models and environmental sensitivity
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class HypothesisTester:
    """
    Statistical hypothesis testing for agricultural prediction models
    Tests the significance of fusion, confidence, and environmental sensitivity
    """
    
    def __init__(self):
        self.results = {}
        self.alpha = 0.05  # Significance level
        
    def test_fusion_significance(
        self,
        predictions_without_fusion: np.ndarray,
        predictions_with_fusion: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Hypothesis 1: Fusion Significance Test
        H0: Fusion does not significantly change prediction score
        H1: Fusion significantly changes prediction score
        Test: Paired t-test
        
        Args:
            predictions_without_fusion: Predictions from individual models
            predictions_with_fusion: Predictions after fusion
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running Hypothesis Test 1: Fusion Significance")
        
        # Ensure same length
        n = min(len(predictions_without_fusion), len(predictions_with_fusion))
        pred_without = predictions_without_fusion[:n]
        pred_with = predictions_with_fusion[:n]
        
        # Calculate differences
        differences = pred_with - pred_without
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(pred_with, pred_without, alternative=alternative)
        
        # Effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Confidence interval for mean difference
        ci = stats.t.interval(
            1 - self.alpha,
            df=n - 1,
            loc=mean_diff,
            scale=stats.sem(differences)
        )
        
        # Decision
        reject_null = p_value < self.alpha
        
        result = {
            'hypothesis': 'Fusion Significance',
            'h0': 'Fusion does not significantly change prediction score',
            'h1': 'Fusion significantly changes prediction score',
            'test_type': 'Paired t-test',
            'n_samples': n,
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Fusion SIGNIFICANTLY changes predictions' if reject_null else 'Fusion does NOT significantly change predictions',
            'mean_without_fusion': float(np.mean(pred_without)),
            'mean_with_fusion': float(np.mean(pred_with)),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'cohens_d': float(cohens_d),
            'effect_size': self._interpret_cohens_d(cohens_d),
            'confidence_interval_95': (float(ci[0]), float(ci[1])),
            'interpretation': self._interpret_fusion_significance(reject_null, mean_diff, cohens_d)
        }
        
        self.results['fusion_significance'] = result
        logger.info(f"Fusion Significance Test: p-value={p_value:.4f}, reject_null={reject_null}")
        
        return result
    
    def test_confidence_increase(
        self,
        confidence_without_fusion: np.ndarray,
        confidence_with_fusion: np.ndarray
    ) -> Dict:
        """
        Hypothesis 2: Prediction Confidence Increase
        H0: Fusion does not increase model confidence
        H1: Fusion increases prediction confidence level
        Test: One-sided paired t-test (greater)
        
        Args:
            confidence_without_fusion: Confidence scores without fusion
            confidence_with_fusion: Confidence scores with fusion
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running Hypothesis Test 2: Confidence Increase")
        
        # Ensure same length
        n = min(len(confidence_without_fusion), len(confidence_with_fusion))
        conf_without = confidence_without_fusion[:n]
        conf_with = confidence_with_fusion[:n]
        
        # One-sided paired t-test (testing if fusion increases confidence)
        t_statistic, p_value = stats.ttest_rel(
            conf_with, 
            conf_without, 
            alternative='greater'
        )
        
        # Calculate statistics
        mean_without = np.mean(conf_without)
        mean_with = np.mean(conf_with)
        mean_increase = mean_with - mean_without
        percent_increase = (mean_increase / mean_without * 100) if mean_without > 0 else 0
        
        # Confidence interval for mean increase
        differences = conf_with - conf_without
        ci = stats.t.interval(
            1 - self.alpha,
            df=n - 1,
            loc=np.mean(differences),
            scale=stats.sem(differences)
        )
        
        # Effect size
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_increase / std_diff if std_diff > 0 else 0
        
        # Decision
        reject_null = p_value < self.alpha
        
        # Additional metrics
        samples_improved = np.sum(conf_with > conf_without)
        improvement_rate = samples_improved / n * 100
        
        result = {
            'hypothesis': 'Prediction Confidence Increase',
            'h0': 'Fusion does not increase model confidence',
            'h1': 'Fusion increases prediction confidence level',
            'test_type': 'One-sided paired t-test (greater)',
            'n_samples': n,
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Fusion SIGNIFICANTLY increases confidence' if reject_null else 'Fusion does NOT significantly increase confidence',
            'mean_confidence_without': float(mean_without),
            'mean_confidence_with': float(mean_with),
            'mean_increase': float(mean_increase),
            'percent_increase': float(percent_increase),
            'cohens_d': float(cohens_d),
            'effect_size': self._interpret_cohens_d(cohens_d),
            'confidence_interval_95': (float(ci[0]), float(ci[1])),
            'samples_improved': int(samples_improved),
            'improvement_rate': float(improvement_rate),
            'interpretation': self._interpret_confidence_increase(reject_null, mean_increase, improvement_rate)
        }
        
        self.results['confidence_increase'] = result
        logger.info(f"Confidence Increase Test: p-value={p_value:.4f}, reject_null={reject_null}")
        
        return result
    
    def test_environmental_sensitivity(
        self,
        prediction_scores: np.ndarray,
        weather_params: pd.DataFrame,
        soil_params: pd.DataFrame = None
    ) -> Dict:
        """
        Hypothesis 3: Environmental Sensitivity
        H0: Weather/soil parameters do not significantly influence prediction output
        H1: Weather/soil parameters significantly influence prediction output
        Test: Multiple correlation analysis and significance testing
        
        Args:
            prediction_scores: Model prediction scores
            weather_params: DataFrame with weather variables (temp, humidity, rainfall, etc.)
            soil_params: Optional DataFrame with soil variables
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running Hypothesis Test 3: Environmental Sensitivity")
        
        n = len(prediction_scores)
        correlations = {}
        p_values = {}
        significant_factors = []
        
        # Test weather parameters
        for col in weather_params.columns:
            if pd.api.types.is_numeric_dtype(weather_params[col]):
                # Pearson correlation
                r, p = stats.pearsonr(prediction_scores[:len(weather_params)], weather_params[col])
                correlations[f'weather_{col}'] = float(r)
                p_values[f'weather_{col}'] = float(p)
                
                if p < self.alpha:
                    significant_factors.append({
                        'factor': f'weather_{col}',
                        'correlation': float(r),
                        'p_value': float(p),
                        'significance': 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
                    })
        
        # Test soil parameters if provided
        if soil_params is not None:
            for col in soil_params.columns:
                if pd.api.types.is_numeric_dtype(soil_params[col]):
                    r, p = stats.pearsonr(prediction_scores[:len(soil_params)], soil_params[col])
                    correlations[f'soil_{col}'] = float(r)
                    p_values[f'soil_{col}'] = float(p)
                    
                    if p < self.alpha:
                        significant_factors.append({
                            'factor': f'soil_{col}',
                            'correlation': float(r),
                            'p_value': float(p),
                            'significance': 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
                        })
        
        # Overall test: Are ANY environmental factors significant?
        reject_null = len(significant_factors) > 0
        
        # Sort by absolute correlation
        significant_factors.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Multiple regression R-squared (if we have multiple factors)
        r_squared = None
        if len(weather_params.columns) > 0:
            from sklearn.linear_model import LinearRegression
            X = weather_params.select_dtypes(include=[np.number]).values[:len(prediction_scores)]
            if X.shape[0] > 0 and X.shape[1] > 0:
                model = LinearRegression()
                model.fit(X, prediction_scores[:len(X)])
                r_squared = float(model.score(X, prediction_scores[:len(X)]))
        
        result = {
            'hypothesis': 'Environmental Sensitivity',
            'h0': 'Weather/soil parameters do not significantly influence prediction output',
            'h1': 'Weather/soil parameters significantly influence prediction output',
            'test_type': 'Pearson correlation analysis',
            'n_samples': n,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Environmental parameters SIGNIFICANTLY influence predictions' if reject_null else 'Environmental parameters do NOT significantly influence predictions',
            'correlations': correlations,
            'p_values': p_values,
            'significant_factors': significant_factors,
            'n_significant_factors': len(significant_factors),
            'r_squared': r_squared,
            'interpretation': self._interpret_environmental_sensitivity(significant_factors, r_squared)
        }
        
        self.results['environmental_sensitivity'] = result
        logger.info(f"Environmental Sensitivity Test: {len(significant_factors)} significant factors found")
        
        return result
    
    def test_ndvi_trend_sensitivity(
        self,
        prediction_scores: np.ndarray,
        ndvi_values: np.ndarray
    ) -> Dict:
        """
        Hypothesis 4: NDVI Trend Sensitivity
        H0: Prediction score does not significantly change with NDVI variation
        H1: Prediction score significantly changes with NDVI variation
        Test: Correlation and linear regression analysis
        
        Args:
            prediction_scores: Model prediction scores
            ndvi_values: NDVI measurements
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running Hypothesis Test 4: NDVI Trend Sensitivity")
        
        # Ensure same length
        n = min(len(prediction_scores), len(ndvi_values))
        pred = prediction_scores[:n]
        ndvi = ndvi_values[:n]
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(pred, ndvi)
        
        # Spearman correlation (for non-linear relationships)
        r_spearman, p_spearman = stats.spearmanr(pred, ndvi)
        
        # Linear regression
        from sklearn.linear_model import LinearRegression
        X = ndvi.reshape(-1, 1)
        y = pred
        
        model = LinearRegression()
        model.fit(X, y)
        
        # R-squared
        r_squared = model.score(X, y)
        
        # Regression coefficients
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # F-test for regression significance
        from scipy.stats import f
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - model.predict(X)) ** 2)
        ss_regression = ss_total - ss_residual
        
        df_regression = 1
        df_residual = n - 2
        
        ms_regression = ss_regression / df_regression
        ms_residual = ss_residual / df_residual if df_residual > 0 else 1e-10
        
        f_statistic = ms_regression / ms_residual
        p_f_test = 1 - f.cdf(f_statistic, df_regression, df_residual)
        
        # Decision
        reject_null = p_pearson < self.alpha
        
        # Confidence interval for slope
        from scipy.stats import t
        se_slope = np.sqrt(ms_residual / np.sum((ndvi - np.mean(ndvi)) ** 2))
        t_critical = t.ppf(1 - self.alpha / 2, df_residual)
        ci_slope = (slope - t_critical * se_slope, slope + t_critical * se_slope)
        
        result = {
            'hypothesis': 'NDVI Trend Sensitivity',
            'h0': 'Prediction score does not significantly change with NDVI variation',
            'h1': 'Prediction score significantly changes with NDVI variation',
            'test_type': 'Correlation and Linear Regression',
            'n_samples': n,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Prediction SIGNIFICANTLY changes with NDVI' if reject_null else 'Prediction does NOT significantly change with NDVI',
            'pearson_r': float(r_pearson),
            'pearson_p_value': float(p_pearson),
            'spearman_r': float(r_spearman),
            'spearman_p_value': float(p_spearman),
            'r_squared': float(r_squared),
            'regression_slope': float(slope),
            'regression_intercept': float(intercept),
            'confidence_interval_slope_95': (float(ci_slope[0]), float(ci_slope[1])),
            'f_statistic': float(f_statistic),
            'f_test_p_value': float(p_f_test),
            'correlation_strength': self._interpret_correlation(r_pearson),
            'interpretation': self._interpret_ndvi_sensitivity(reject_null, r_pearson, slope, r_squared)
        }
        
        self.results['ndvi_sensitivity'] = result
        logger.info(f"NDVI Sensitivity Test: r={r_pearson:.3f}, p-value={p_pearson:.4f}, reject_null={reject_null}")
        
        return result
    
    def run_all_tests(
        self,
        predictions_without_fusion: np.ndarray,
        predictions_with_fusion: np.ndarray,
        confidence_without_fusion: np.ndarray,
        confidence_with_fusion: np.ndarray,
        prediction_scores: np.ndarray,
        ndvi_values: np.ndarray,
        weather_params: pd.DataFrame,
        soil_params: pd.DataFrame = None
    ) -> Dict:
        """
        Run all four hypothesis tests
        
        Returns:
            Dictionary with all test results
        """
        logger.info("Running all hypothesis tests...")
        
        # Test 1: Fusion Significance
        test1 = self.test_fusion_significance(
            predictions_without_fusion,
            predictions_with_fusion
        )
        
        # Test 2: Confidence Increase
        test2 = self.test_confidence_increase(
            confidence_without_fusion,
            confidence_with_fusion
        )
        
        # Test 3: Environmental Sensitivity
        test3 = self.test_environmental_sensitivity(
            prediction_scores,
            weather_params,
            soil_params
        )
        
        # Test 4: NDVI Sensitivity
        test4 = self.test_ndvi_trend_sensitivity(
            prediction_scores,
            ndvi_values
        )
        
        summary = {
            'fusion_significance': test1,
            'confidence_increase': test2,
            'environmental_sensitivity': test3,
            'ndvi_sensitivity': test4,
            'overall_summary': self._generate_overall_summary()
        }
        
        logger.info("All hypothesis tests completed")
        return summary
    
    def generate_report(self, output_path: str = None) -> pd.DataFrame:
        """Generate a summary report of all hypothesis tests"""
        if not self.results:
            logger.warning("No test results available. Run tests first.")
            return None
        
        report_data = []
        for test_name, test_result in self.results.items():
            report_data.append({
                'Test': test_result['hypothesis'],
                'H0': test_result['h0'],
                'H1': test_result['h1'],
                'Test Type': test_result['test_type'],
                'p-value': test_result.get('p_value', test_result.get('pearson_p_value', 'N/A')),
                'Reject H0': test_result['reject_null'],
                'Conclusion': test_result['conclusion']
            })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Report saved to {output_path}")
        
        return report_df
    
    # Helper methods for interpretation
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "Weak"
        elif abs_r < 0.7:
            return "Moderate"
        else:
            return "Strong"
    
    def _interpret_fusion_significance(self, reject_null: bool, mean_diff: float, cohens_d: float) -> str:
        """Generate interpretation for fusion significance test"""
        if not reject_null:
            return "Fusion does not produce statistically significant changes in predictions. The difference could be due to random chance."
        else:
            direction = "increases" if mean_diff > 0 else "decreases"
            magnitude = self._interpret_cohens_d(cohens_d)
            return f"Fusion significantly {direction} prediction scores with a {magnitude.lower()} effect size (Cohen's d = {cohens_d:.3f}). This suggests fusion meaningfully improves the model."
    
    def _interpret_confidence_increase(self, reject_null: bool, mean_increase: float, improvement_rate: float) -> str:
        """Generate interpretation for confidence increase test"""
        if not reject_null:
            return "Fusion does not significantly increase model confidence. The observed increase could be due to random variation."
        else:
            return f"Fusion significantly increases prediction confidence by an average of {mean_increase:.3f} ({improvement_rate:.1f}% of samples showed improvement). This indicates the model is more certain when using fused data."
    
    def _interpret_environmental_sensitivity(self, significant_factors: List, r_squared: float) -> str:
        """Generate interpretation for environmental sensitivity test"""
        if len(significant_factors) == 0:
            return "No environmental factors show significant correlation with predictions. The model may not be sensitive to environmental variations."
        else:
            top_factor = significant_factors[0]
            interpretation = f"Found {len(significant_factors)} significant environmental factors. "
            interpretation += f"The strongest predictor is {top_factor['factor']} (r={top_factor['correlation']:.3f}, p={top_factor['p_value']:.4f}). "
            if r_squared:
                interpretation += f"Environmental factors collectively explain {r_squared*100:.1f}% of prediction variance."
            return interpretation
    
    def _interpret_ndvi_sensitivity(self, reject_null: bool, r: float, slope: float, r_squared: float) -> str:
        """Generate interpretation for NDVI sensitivity test"""
        if not reject_null:
            return "NDVI does not show significant relationship with prediction scores. The model may not be responding appropriately to vegetation health indicators."
        else:
            direction = "increases" if slope > 0 else "decreases"
            strength = self._interpret_correlation(r)
            return f"Prediction scores show {strength.lower()} {direction} with NDVI (r={r:.3f}, R²={r_squared:.3f}). For every 0.1 increase in NDVI, predictions change by {slope*0.1:.3f}. This confirms the model appropriately responds to vegetation health."
    
    def _generate_overall_summary(self) -> Dict:
        """Generate overall summary of all tests"""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        rejected_nulls = sum(1 for r in self.results.values() if r['reject_null'])
        
        return {
            'total_tests': total_tests,
            'significant_results': rejected_nulls,
            'significance_rate': rejected_nulls / total_tests if total_tests > 0 else 0,
            'model_validity': 'High' if rejected_nulls >= 3 else 'Moderate' if rejected_nulls >= 2 else 'Low'
        }


def generate_synthetic_test_data(n_samples: int = 100, scenario: str = 'realistic') -> Dict:
    """
    Generate synthetic data for hypothesis testing
    Useful for validation and demonstration
    
    Args:
        n_samples: Number of samples to generate
        scenario: 'realistic' (mixed results), 'significant' (all significant), 'null' (all null)
    """
    # Use time-based seed for variability across runs
    import time
    seed = int(time.time() * 1000) % 100000
    np.random.seed(seed)
    
    # Environmental parameters
    temperature = np.random.normal(25, 5, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    rainfall = np.random.exponential(10, n_samples)
    
    # NDVI values (0 to 1)
    ndvi = np.random.beta(8, 2, n_samples)
    
    # Adjust effects based on scenario - make realistic scenario truly variable
    if scenario == 'realistic':
        # Randomly decide which hypotheses will be significant
        # This creates natural variation - sometimes fusion helps, sometimes it doesn't
        fusion_significant = np.random.random() > 0.4  # 60% chance of significance
        confidence_significant = np.random.random() > 0.3  # 70% chance
        
        if fusion_significant:
            # Noticeable but not huge fusion effect
            fusion_effect = np.random.uniform(0.03, 0.08, n_samples)
        else:
            # Minimal fusion effect - null hypothesis likely true
            fusion_effect = np.random.normal(0, 0.015, n_samples)  # Near zero
        
        if confidence_significant:
            # Modest confidence boost
            confidence_boost = np.random.uniform(0.03, 0.08, n_samples)
        else:
            # Minimal or no confidence boost
            confidence_boost = np.random.uniform(-0.01, 0.02, n_samples)
            
    elif scenario == 'null':
        # Minimal to no effect (null hypothesis is true)
        fusion_effect = np.random.normal(0, 0.01, n_samples)  # Almost no effect
        confidence_boost = np.random.uniform(-0.02, 0.02, n_samples)  # No real boost
    else:  # 'significant'
        # Strong effects (for demonstration)
        fusion_effect = np.random.uniform(0.08, 0.15, n_samples)
        confidence_boost = np.random.uniform(0.08, 0.15, n_samples)
    
    # Predictions without fusion (based on NDVI only)
    predictions_without = 0.3 + 0.6 * ndvi + np.random.normal(0, 0.05, n_samples)
    predictions_without = np.clip(predictions_without, 0, 1)
    
    # Predictions with fusion (incorporate environmental factors with variable effect)
    # Make environmental effects also variable
    if scenario == 'realistic':
        env_strength = np.random.uniform(0.02, 0.06)  # Variable environmental influence
    else:
        env_strength = 0.05
        
    temp_effect = (temperature - 20) / 30 * env_strength
    humidity_effect = (humidity - 50) / 50 * (env_strength * 0.6)
    
    predictions_with = predictions_without + fusion_effect + temp_effect + humidity_effect + np.random.normal(0, 0.04, n_samples)
    predictions_with = np.clip(predictions_with, 0, 1)
    
    # Confidence scores
    confidence_without = np.abs(predictions_without - 0.5) * 2  # Higher confidence at extremes
    confidence_with = confidence_without + confidence_boost
    confidence_with = np.clip(confidence_with, 0, 1)
    
    # Weather dataframe
    weather_df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    })
    
    # Soil dataframe
    soil_df = pd.DataFrame({
        'ph': np.random.normal(6.5, 0.5, n_samples),
        'nitrogen': np.random.normal(50, 10, n_samples),
        'phosphorus': np.random.normal(30, 8, n_samples)
    })
    
    return {
        'predictions_without_fusion': predictions_without,
        'predictions_with_fusion': predictions_with,
        'confidence_without_fusion': confidence_without,
        'confidence_with_fusion': confidence_with,
        'prediction_scores': predictions_with,
        'ndvi_values': ndvi,
        'weather_params': weather_df,
        'soil_params': soil_df
    }


if __name__ == "__main__":
    # Example usage
    logger.info("Running hypothesis testing example...")
    
    # Generate test data
    data = generate_synthetic_test_data(n_samples=150)
    
    # Initialize tester
    tester = HypothesisTester()
    
    # Run all tests
    results = tester.run_all_tests(**data)
    
    # Generate report
    report = tester.generate_report('hypothesis_test_results.csv')
    print("\n" + "="*80)
    print("HYPOTHESIS TEST RESULTS")
    print("="*80)
    print(report.to_string(index=False))
    
    # Print detailed results
    for test_name, test_result in results.items():
        if test_name != 'overall_summary':
            print(f"\n{'='*80}")
            print(f"TEST: {test_result['hypothesis']}")
            print(f"{'='*80}")
            print(f"H0: {test_result['h0']}")
            print(f"H1: {test_result['h1']}")
            print(f"p-value: {test_result.get('p_value', test_result.get('pearson_p_value', 'N/A')):.4f}")
            print(f"Conclusion: {test_result['conclusion']}")
            print(f"Interpretation: {test_result['interpretation']}")
