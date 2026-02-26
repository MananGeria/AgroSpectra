"""
Visualization Module for Hypothesis Testing Results
Creates comprehensive charts and plots for statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
from loguru import logger


class HypothesisVisualizer:
    """Generate visualizations for hypothesis test results"""
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_fusion_comparison(
        self,
        predictions_without: np.ndarray,
        predictions_with: np.ndarray,
        test_result: Dict
    ):
        """
        Visualize Hypothesis 1: Fusion Significance
        Creates comparison plots and distribution analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Prediction Distributions',
                'Before vs After Fusion',
                'Difference Distribution',
                'Paired Comparison (First 30 samples)'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Distribution comparison
        fig.add_trace(
            go.Histogram(
                x=predictions_without,
                name='Without Fusion',
                opacity=0.7,
                marker_color='lightblue',
                nbinsx=30
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=predictions_with,
                name='With Fusion',
                opacity=0.7,
                marker_color='darkblue',
                nbinsx=30
            ),
            row=1, col=1
        )
        
        # 2. Scatter plot: Before vs After
        fig.add_trace(
            go.Scatter(
                x=predictions_without,
                y=predictions_with,
                mode='markers',
                marker=dict(
                    color=predictions_with - predictions_without,
                    colorscale='RdYlGn',
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Difference", x=1.15)
                ),
                name='Predictions',
                showlegend=False
            ),
            row=1, col=2
        )
        # Add diagonal line
        min_val = min(predictions_without.min(), predictions_with.min())
        max_val = max(predictions_without.max(), predictions_with.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='No Change Line',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Difference distribution
        differences = predictions_with - predictions_without
        fig.add_trace(
            go.Histogram(
                x=differences,
                marker_color='green' if test_result['mean_difference'] > 0 else 'red',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=1
        )
        # Add mean line
        fig.add_vline(
            x=test_result['mean_difference'],
            line_dash="dash",
            line_color="black",
            annotation_text=f"Mean: {test_result['mean_difference']:.3f}",
            row=2, col=1
        )
        
        # 4. Paired line plot (first 30 samples)
        n_display = min(30, len(predictions_without))
        indices = np.arange(n_display)
        
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=predictions_without[:n_display],
                mode='lines+markers',
                name='Without Fusion',
                line=dict(color='lightblue', width=2)
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=predictions_with[:n_display],
                mode='lines+markers',
                name='With Fusion',
                line=dict(color='darkblue', width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Prediction Score", row=1, col=1)
        fig.update_xaxes(title_text="Without Fusion", row=1, col=2)
        fig.update_xaxes(title_text="Difference (With - Without)", row=2, col=1)
        fig.update_xaxes(title_text="Sample Index", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="With Fusion", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Prediction Score", row=2, col=2)
        
        # Add test result annotation
        p_val = test_result['p_value']
        conclusion = test_result['conclusion']
        
        fig.add_annotation(
            text=f"<b>p-value: {p_val:.4f}</b><br>{conclusion}<br>Effect Size (Cohen's d): {test_result['cohens_d']:.3f} ({test_result['effect_size']})",
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=12),
            bgcolor="lightyellow",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text=f"<b>Hypothesis 1: Fusion Significance Test</b>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_confidence_increase(
        self,
        confidence_without: np.ndarray,
        confidence_with: np.ndarray,
        test_result: Dict
    ):
        """
        Visualize Hypothesis 2: Confidence Increase
        Shows confidence improvements after fusion
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confidence Score Distributions',
                'Individual Sample Changes',
                'Improvement Distribution',
                'Cumulative Improvement'
            )
        )
        
        # 1. Box plots for confidence scores
        fig.add_trace(
            go.Box(
                y=confidence_without,
                name='Without Fusion',
                marker_color='lightcoral',
                boxmean='sd'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(
                y=confidence_with,
                name='With Fusion',
                marker_color='lightgreen',
                boxmean='sd'
            ),
            row=1, col=1
        )
        
        # 2. Individual sample changes
        improvements = confidence_with - confidence_without
        colors = ['green' if i > 0 else 'red' if i < 0 else 'gray' for i in improvements]
        
        fig.add_trace(
            go.Bar(
                x=np.arange(len(improvements)),
                y=improvements,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Improvement distribution
        fig.add_trace(
            go.Histogram(
                x=improvements,
                marker_color='teal',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            annotation_text="No Change",
            row=2, col=1
        )
        fig.add_vline(
            x=test_result['mean_increase'],
            line_dash="solid",
            line_color="red",
            annotation_text=f"Mean: +{test_result['mean_increase']:.3f}",
            row=2, col=1
        )
        
        # 4. Cumulative improvement
        sorted_improvements = np.sort(improvements)
        cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements) * 100
        
        fig.add_trace(
            go.Scatter(
                x=sorted_improvements,
                y=cumulative,
                mode='lines',
                fill='tozeroy',
                line=dict(color='purple', width=3),
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            annotation_text=f"{test_result['improvement_rate']:.1f}% improved",
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Model Type", row=1, col=1)
        fig.update_xaxes(title_text="Sample Index", row=1, col=2)
        fig.update_xaxes(title_text="Confidence Change", row=2, col=1)
        fig.update_xaxes(title_text="Confidence Change", row=2, col=2)
        
        fig.update_yaxes(title_text="Confidence Score", row=1, col=1)
        fig.update_yaxes(title_text="Change in Confidence", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative %", row=2, col=2)
        
        # Add test result
        fig.add_annotation(
            text=f"<b>p-value: {test_result['p_value']:.4f}</b><br>{test_result['conclusion']}<br>Mean Increase: +{test_result['mean_increase']:.3f} ({test_result['percent_increase']:.1f}%)",
            xref="paper", yref="paper",
            x=0.5, y=1.12,
            showarrow=False,
            font=dict(size=12),
            bgcolor="lightgreen",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text="<b>Hypothesis 2: Prediction Confidence Increase Test</b>",
            height=800
        )
        
        return fig
    
    def plot_environmental_sensitivity(
        self,
        test_result: Dict,
        prediction_scores: np.ndarray,
        weather_params: pd.DataFrame
    ):
        """
        Visualize Hypothesis 3: Environmental Sensitivity
        Shows correlations with environmental factors
        """
        significant_factors = test_result['significant_factors']
        
        # Create correlation heatmap data
        correlations = test_result['correlations']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Correlation Heatmap',
                'Top Factor Scatter Plot',
                'P-values by Factor',
                'Environmental Factor Contributions'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        # 1. Correlation heatmap
        factor_names = list(correlations.keys())
        corr_values = list(correlations.values())
        p_values = [test_result['p_values'][f] for f in factor_names]
        
        # Create text annotations for heatmap
        text_annotations = [
            [f"r={corr_values[i]:.3f}<br>p={p_values[i]:.4f}" for i in range(len(factor_names))]
        ]
        
        fig.add_trace(
            go.Heatmap(
                z=[corr_values],
                x=factor_names,
                y=['Correlation'],
                colorscale='RdBu',
                zmid=0,
                text=text_annotations,
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation", x=1.15)
            ),
            row=1, col=1
        )
        
        # 2. Top factor scatter plot
        if significant_factors:
            top_factor = significant_factors[0]['factor']
            factor_key = top_factor.replace('weather_', '').replace('soil_', '')
            
            if factor_key in weather_params.columns:
                x_data = weather_params[factor_key].values[:len(prediction_scores)]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=prediction_scores,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=prediction_scores,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Prediction", x=1.3)
                        ),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add trend line
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                X = x_data.reshape(-1, 1)
                model.fit(X, prediction_scores)
                y_pred = model.predict(X)
                
                # Sort for smooth line
                sort_idx = np.argsort(x_data)
                fig.add_trace(
                    go.Scatter(
                        x=x_data[sort_idx],
                        y=y_pred[sort_idx],
                        mode='lines',
                        line=dict(color='red', width=3, dash='dash'),
                        name=f'Trend (r={significant_factors[0]["correlation"]:.3f})'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text=top_factor, row=1, col=2)
        
        # 3. P-values bar chart
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        fig.add_trace(
            go.Bar(
                x=factor_names,
                y=p_values,
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="black",
            annotation_text="α=0.05",
            row=2, col=1
        )
        
        # 4. Absolute correlation values
        abs_corr = [abs(c) for c in corr_values]
        colors_strength = ['darkgreen' if c > 0.7 else 'green' if c > 0.4 else 'orange' for c in abs_corr]
        
        fig.add_trace(
            go.Bar(
                x=factor_names,
                y=abs_corr,
                marker_color=colors_strength,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Prediction Score", row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        fig.update_yaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="Prediction Score", row=1, col=2)
        fig.update_yaxes(title_text="p-value", row=2, col=1, type='log')
        fig.update_yaxes(title_text="|Correlation|", row=2, col=2)
        
        # Add test result
        n_sig = len(significant_factors)
        fig.add_annotation(
            text=f"<b>Found {n_sig} significant factors (p < 0.05)</b><br>{test_result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=1.12,
            showarrow=False,
            font=dict(size=12),
            bgcolor="lightblue",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text="<b>Hypothesis 3: Environmental Sensitivity Test</b>",
            height=800
        )
        
        return fig
    
    def plot_ndvi_sensitivity(
        self,
        prediction_scores: np.ndarray,
        ndvi_values: np.ndarray,
        test_result: Dict
    ):
        """
        Visualize Hypothesis 4: NDVI Trend Sensitivity
        Shows relationship between NDVI and predictions
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'NDVI vs Prediction Score (with Regression)',
                'Residual Plot',
                'NDVI Distribution',
                'Prediction Score Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'histogram'}]
            ]
        )
        
        # 1. Scatter plot with regression line
        fig.add_trace(
            go.Scatter(
                x=ndvi_values,
                y=prediction_scores,
                mode='markers',
                marker=dict(
                    size=8,
                    color=prediction_scores,
                    colorscale='Greens',
                    showscale=True,
                    colorbar=dict(title="Prediction", x=1.15)
                ),
                name='Data Points'
            ),
            row=1, col=1
        )
        
        # Add regression line
        slope = test_result['regression_slope']
        intercept = test_result['regression_intercept']
        x_line = np.linspace(ndvi_values.min(), ndvi_values.max(), 100)
        y_line = slope * x_line + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name=f'y = {slope:.3f}x + {intercept:.3f}'
            ),
            row=1, col=1
        )
        
        # Add confidence interval
        from scipy import stats as sp_stats
        # Calculate prediction intervals
        y_pred = slope * ndvi_values + intercept
        residuals = prediction_scores - y_pred
        std_residuals = np.std(residuals)
        
        y_upper = y_line + 1.96 * std_residuals
        y_lower = y_line - 1.96 * std_residuals
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% CI',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Residual plot
        fig.add_trace(
            go.Scatter(
                x=ndvi_values,
                y=residuals,
                mode='markers',
                marker=dict(
                    size=6,
                    color='purple',
                    opacity=0.6
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            row=1, col=2
        )
        
        # 3. NDVI distribution
        fig.add_trace(
            go.Histogram(
                x=ndvi_values,
                marker_color='green',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Prediction distribution
        fig.add_trace(
            go.Histogram(
                x=prediction_scores,
                marker_color='blue',
                opacity=0.7,
                nbinsx=30,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="NDVI", row=1, col=1)
        fig.update_xaxes(title_text="NDVI", row=1, col=2)
        fig.update_xaxes(title_text="NDVI", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Score", row=2, col=2)
        
        fig.update_yaxes(title_text="Prediction Score", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Add test result
        r = test_result['pearson_r']
        r_sq = test_result['r_squared']
        p_val = test_result['pearson_p_value']
        
        fig.add_annotation(
            text=f"<b>r = {r:.3f}, R² = {r_sq:.3f}, p = {p_val:.4f}</b><br>{test_result['conclusion']}<br>Correlation: {test_result['correlation_strength']}",
            xref="paper", yref="paper",
            x=0.5, y=1.12,
            showarrow=False,
            font=dict(size=12),
            bgcolor="lightgreen" if test_result['reject_null'] else "lightyellow",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text="<b>Hypothesis 4: NDVI Trend Sensitivity Test</b>",
            height=800
        )
        
        return fig
    
    def create_summary_dashboard(self, all_results: Dict):
        """Create comprehensive summary dashboard of all tests"""
        fig = go.Figure()
        
        # Extract key metrics
        tests = []
        p_values = []
        rejected = []
        
        test_mapping = {
            'fusion_significance': 'Fusion Significance',
            'confidence_increase': 'Confidence Increase',
            'environmental_sensitivity': 'Environmental Sensitivity',
            'ndvi_sensitivity': 'NDVI Sensitivity'
        }
        
        for key, name in test_mapping.items():
            if key in all_results:
                result = all_results[key]
                tests.append(name)
                p_val = result.get('p_value', result.get('pearson_p_value', 1.0))
                p_values.append(p_val)
                rejected.append(result['reject_null'])
        
        # Create bar chart for p-values
        colors = ['green' if r else 'red' for r in rejected]
        
        fig.add_trace(
            go.Bar(
                x=tests,
                y=p_values,
                marker_color=colors,
                text=[f"p={p:.4f}" for p in p_values],
                textposition='outside'
            )
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="black",
            annotation_text="α = 0.05 (Significance Threshold)"
        )
        
        fig.update_layout(
            title="<b>Hypothesis Testing Summary - All Tests</b>",
            xaxis_title="Hypothesis Test",
            yaxis_title="p-value",
            yaxis_type="log",
            height=500,
            showlegend=False
        )
        
        # Add summary annotation
        n_rejected = sum(rejected)
        summary_text = f"<b>{n_rejected}/{len(tests)} hypotheses rejected H0</b><br>"
        summary_text += f"Model Validity: {all_results.get('overall_summary', {}).get('model_validity', 'Unknown')}"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=14),
            bgcolor="lightgreen" if n_rejected >= 3 else "lightyellow",
            bordercolor="black",
            borderwidth=2
        )
        
        return fig
