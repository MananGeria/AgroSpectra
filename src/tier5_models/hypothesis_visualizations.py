"""
Visualization Module for Hypothesis Testing Results
Creates comprehensive charts and plots for statistical analysis.
Each plot embeds the formula used AND all computed statistical values
directly on the graph for transparency and educational clarity.
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


# ──────────────────────────────────────────────
#  Module-level helper: format p-values neatly
# ──────────────────────────────────────────────
def _fmt_p(p) -> str:
    """Return a clean string for a p-value (scientific notation when very small)."""
    if not isinstance(p, (int, float)):
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


class HypothesisVisualizer:
    """Generate visualizations for hypothesis test results.
    
    Every plot method annotates:
      • The statistical formula being applied
      • All computed test statistics (t, r, R², F, p, CI, Cohen's d …)
      • A clear REJECT / FAIL TO REJECT banner
    """

    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    # ══════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 1 — Fusion Significance  (Paired t-test, two-sided)
    # ══════════════════════════════════════════════════════════════════
    def plot_fusion_comparison(
        self,
        predictions_without: np.ndarray,
        predictions_with: np.ndarray,
        test_result: Dict
    ):
        """
        H1: μ_with ≠ μ_without
        FORMULA: t = d̄ / (s_d / √n)
          d̄   = mean of paired differences
          s_d  = std dev of paired differences
          n    = number of paired observations
        Also: Cohen's d = d̄ / s_d  (effect size)
        """
        # ── pull values from result dict ──────────────────────────────
        t_stat    = test_result.get('t_statistic', float('nan'))
        p_val     = test_result.get('p_value', float('nan'))
        mean_diff = test_result.get('mean_difference', float('nan'))
        std_diff  = test_result.get('std_difference', float('nan'))
        cohens_d  = test_result.get('cohens_d', float('nan'))
        effect    = test_result.get('effect_size', '—')
        ci        = test_result.get('confidence_interval_95', [float('nan'), float('nan')])
        mean_wo   = test_result.get('mean_without_fusion', float('nan'))
        mean_w    = test_result.get('mean_with_fusion', float('nan'))
        reject    = test_result.get('reject_null', False)
        n         = len(predictions_without)

        decision_label = "✅ REJECT H₀  (significant difference)" if reject else "❌ FAIL TO REJECT H₀  (no significant difference)"
        decision_color = "rgba(0,180,0,0.15)" if reject else "rgba(220,0,0,0.12)"

        # ── formula + stats banner ────────────────────────────────────
        formula_text = (
            "<b>FORMULA (Paired t-test, two-sided):</b>  "
            "t = d̄ / (s_d / √n)<br>"
            f"  n = {n}  |  "
            f"d̄ (mean diff) = {mean_diff:+.4f}  |  "
            f"s_d = {std_diff:.4f}  |  "
            f"t = {t_stat:.4f}  |  "
            f"p = {_fmt_p(p_val)}  |  α = 0.05<br>"
            f"  Cohen's d = {cohens_d:.4f} ({effect} effect)  |  "
            f"95 % CI = [{ci[0]:.4f}, {ci[1]:.4f}]  |  "
            f"μ_without = {mean_wo:.4f}  |  "
            f"μ_with = {mean_w:.4f}<br>"
            f"  <b>{decision_label}</b>"
        )

        # ── subplots ──────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Prediction Distributions',
                'Before vs After Fusion',
                'Paired Differences  (d = with − without)',
                'Paired Comparison (first 30 samples)'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.12
        )

        # 1 ── Distribution comparison
        fig.add_trace(
            go.Histogram(x=predictions_without, name='Without Fusion',
                         opacity=0.70, marker_color='lightblue', nbinsx=30),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=predictions_with, name='With Fusion',
                         opacity=0.70, marker_color='darkblue', nbinsx=30),
            row=1, col=1
        )
        # mean lines
        fig.add_vline(x=mean_wo, line_dash='dash', line_color='steelblue',
                      annotation_text=f'μ={mean_wo:.3f}', row=1, col=1)
        fig.add_vline(x=mean_w, line_dash='solid', line_color='navy',
                      annotation_text=f'μ={mean_w:.3f}', row=1, col=1)

        # 2 ── Scatter: Before vs After
        diffs_colour = predictions_with - predictions_without
        fig.add_trace(
            go.Scatter(
                x=predictions_without, y=predictions_with, mode='markers',
                marker=dict(color=diffs_colour, colorscale='RdYlGn', size=8,
                            showscale=True, colorbar=dict(title='Δ', x=1.15)),
                name='Samples', showlegend=False
            ),
            row=1, col=2
        )
        mn = min(predictions_without.min(), predictions_with.min())
        mx = max(predictions_without.max(), predictions_with.max())
        fig.add_trace(
            go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                       line=dict(color='red', dash='dash'),
                       name='No Change', showlegend=False),
            row=1, col=2
        )
        # annotate slope of actual change
        fig.add_annotation(
            x=0.98, y=0.05, xref='x2 domain', yref='y2 domain',
            text=f'Δμ = {mean_diff:+.3f}',
            showarrow=False, font=dict(size=12, color='darkgreen' if mean_diff > 0 else 'red'),
            bgcolor='white', bordercolor='gray', borderwidth=1
        )

        # 3 ── Difference distribution
        differences = predictions_with - predictions_without
        fig.add_trace(
            go.Histogram(
                x=differences,
                marker_color='seagreen' if mean_diff >= 0 else 'crimson',
                opacity=0.75, nbinsx=30, showlegend=False
            ),
            row=2, col=1
        )
        # CI shading via shapes (added to layout later)
        fig.add_vline(x=0, line_dash='dash', line_color='gray',
                      annotation_text='H₀: d̄=0', row=2, col=1)
        fig.add_vline(x=mean_diff, line_dash='solid', line_color='black',
                      annotation_text=f'd̄={mean_diff:+.3f}', row=2, col=1)
        fig.add_vline(x=ci[0], line_dash='dot', line_color='darkorange',
                      annotation_text=f'CI lo={ci[0]:.3f}', row=2, col=1)
        fig.add_vline(x=ci[1], line_dash='dot', line_color='darkorange',
                      annotation_text=f'CI hi={ci[1]:.3f}', row=2, col=1)
        # formula annotation inside subplot 3
        fig.add_annotation(
            x=0.98, y=0.95, xref='x3 domain', yref='y3 domain',
            text=(f't = {t_stat:.3f}<br>p = {_fmt_p(p_val)}<br>'
                  f"Cohen's d = {cohens_d:.3f}"),
            showarrow=False, align='right',
            font=dict(size=11), bgcolor='lightyellow',
            bordercolor='black', borderwidth=1
        )

        # 4 ── Paired line plot
        n_disp = min(30, n)
        idx = np.arange(n_disp)
        fig.add_trace(
            go.Scatter(x=idx, y=predictions_without[:n_disp], mode='lines+markers',
                       name='Without Fusion', line=dict(color='lightblue', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=idx, y=predictions_with[:n_disp], mode='lines+markers',
                       name='With Fusion', line=dict(color='darkblue', width=2)),
            row=2, col=2
        )

        # ── axis labels ───────────────────────────────────────────────
        fig.update_xaxes(title_text='Prediction Score', row=1, col=1)
        fig.update_xaxes(title_text='Without Fusion', row=1, col=2)
        fig.update_xaxes(title_text='Difference  (with − without)', row=2, col=1)
        fig.update_xaxes(title_text='Sample Index', row=2, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='With Fusion', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Prediction Score', row=2, col=2)

        # ── title + formula banner ────────────────────────────────────
        fig.update_layout(
            title_text='<b>Hypothesis 1: Fusion Significance Test</b>',
            height=870,
            showlegend=True,
            paper_bgcolor=decision_color,
            margin=dict(t=220)
        )
        fig.add_annotation(
            text=formula_text,
            xref='paper', yref='paper',
            x=0.5, y=1.0,
            xanchor='center', yanchor='bottom',
            showarrow=False,
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='navy', borderwidth=1,
            align='left'
        )
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 2 — Confidence Increase  (One-sided paired t-test)
    # ══════════════════════════════════════════════════════════════════
    def plot_confidence_increase(
        self,
        confidence_without: np.ndarray,
        confidence_with: np.ndarray,
        test_result: Dict
    ):
        """
        H2: μ_with > μ_without  (one-sided / right-tailed)
        FORMULA: t = d̄ / (s_d / √n),  p = P(T > t_obs)
          d̄   = mean of paired differences
          s_d  = std dev of differences
          n    = number of paired observations
        """
        # ── pull values ───────────────────────────────────────────────
        t_stat   = test_result.get('t_statistic', float('nan'))
        p_val    = test_result.get('p_value', float('nan'))
        mean_inc = test_result.get('mean_increase', float('nan'))
        pct_inc  = test_result.get('percent_increase', float('nan'))
        imp_rate = test_result.get('improvement_rate', float('nan'))
        n_imp    = test_result.get('samples_improved', 0)
        cohens_d = test_result.get('cohens_d', float('nan'))
        ci       = test_result.get('confidence_interval_95', [float('nan'), float('nan')])
        mean_wo  = np.mean(confidence_without)
        mean_w   = np.mean(confidence_with)
        reject   = test_result.get('reject_null', False)
        n        = len(confidence_without)

        decision_label = "✅ REJECT H₀  (confidence DID increase)" if reject else "❌ FAIL TO REJECT H₀  (no significant increase)"
        decision_color = "rgba(0,180,0,0.12)" if reject else "rgba(220,0,0,0.10)"

        formula_text = (
            "<b>FORMULA (One-sided Paired t-test — right tail):</b>  "
            "H₁: μ_with > μ_without<br>"
            "  t = d̄ / (s_d / √n)   p = P(T_{n-1} > t_obs)<br>"
            f"  n = {n}  |  "
            f"μ_without = {mean_wo:.4f}  |  "
            f"μ_with = {mean_w:.4f}  |  "
            f"d̄ = {mean_inc:+.4f} ({pct_inc:+.2f} %)  |  "
            f"t = {t_stat:.4f}  |  p = {_fmt_p(p_val)}  |  α = 0.05<br>"
            f"  Cohen's d = {cohens_d:.4f}  |  "
            f"95 % CI = [{ci[0]:.4f}, {ci[1]:.4f}]  |  "
            f"{n_imp}/{n} samples improved ({imp_rate:.1f} %)<br>"
            f"  <b>{decision_label}</b>"
        )

        # ── subplots ──────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confidence Score Distributions (Box)',
                'Per-sample Confidence Change',
                'Distribution of Changes',
                'Cumulative % Improved'
            ),
            vertical_spacing=0.20,
            horizontal_spacing=0.12
        )

        # 1 ── Box plots
        fig.add_trace(
            go.Box(y=confidence_without, name='Without Fusion',
                   marker_color='lightcoral', boxmean='sd'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=confidence_with, name='With Fusion',
                   marker_color='lightgreen', boxmean='sd'),
            row=1, col=1
        )
        # annotate means inside subplot
        fig.add_annotation(
            x=0.5, y=0.95, xref='x1 domain', yref='y1 domain',
            text=f'μ_without={mean_wo:.3f}<br>μ_with={mean_w:.3f}<br>Δ={mean_inc:+.3f}',
            showarrow=False, font=dict(size=11), bgcolor='lightyellow',
            bordercolor='black', borderwidth=1, align='center'
        )

        # 2 ── Per-sample bar
        improvements = confidence_with - confidence_without
        colors = ['seagreen' if i > 0 else 'crimson' if i < 0 else 'gray'
                  for i in improvements]
        fig.add_trace(
            go.Bar(x=np.arange(len(improvements)), y=improvements,
                   marker_color=colors, showlegend=False),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=2)
        # stat box inside subplot 2
        fig.add_annotation(
            x=0.98, y=0.95, xref='x2 domain', yref='y2 domain',
            text=f't = {t_stat:.3f}<br>p = {_fmt_p(p_val)}<br>{n_imp}/{n} improved',
            showarrow=False, align='right',
            font=dict(size=11), bgcolor='lightyellow',
            bordercolor='black', borderwidth=1
        )

        # 3 ── Improvement histogram
        fig.add_trace(
            go.Histogram(x=improvements, marker_color='teal',
                         opacity=0.75, nbinsx=30, showlegend=False),
            row=2, col=1
        )
        fig.add_vline(x=0, line_dash='dash', line_color='gray',
                      annotation_text='H₀: d̄=0', row=2, col=1)
        fig.add_vline(x=mean_inc, line_dash='solid', line_color='black',
                      annotation_text=f'd̄={mean_inc:+.3f}', row=2, col=1)
        fig.add_vline(x=ci[0], line_dash='dot', line_color='darkorange',
                      annotation_text=f'CI lo={ci[0]:.3f}', row=2, col=1)
        fig.add_vline(x=ci[1], line_dash='dot', line_color='darkorange',
                      annotation_text=f'CI hi={ci[1]:.3f}', row=2, col=1)
        # Cohen's d box inside subplot 3
        fig.add_annotation(
            x=0.02, y=0.95, xref='x3 domain', yref='y3 domain',
            text=f"Cohen's d={cohens_d:.3f}<br>({test_result.get('effect_size','—')} effect)",
            showarrow=False, align='left',
            font=dict(size=11), bgcolor='lightyellow',
            bordercolor='black', borderwidth=1
        )

        # 4 ── Cumulative CDF
        sorted_imp = np.sort(improvements)
        cumulative = np.arange(1, len(sorted_imp) + 1) / len(sorted_imp) * 100
        fig.add_trace(
            go.Scatter(x=sorted_imp, y=cumulative, mode='lines',
                       fill='tozeroy', line=dict(color='purple', width=3),
                       showlegend=False),
            row=2, col=2
        )
        fig.add_vline(x=0, line_dash='dash', line_color='black',
                      annotation_text=f'{imp_rate:.1f}% improved',
                      row=2, col=2)

        # ── axis labels ───────────────────────────────────────────────
        fig.update_xaxes(title_text='Model Type', row=1, col=1)
        fig.update_xaxes(title_text='Sample Index', row=1, col=2)
        fig.update_xaxes(title_text='Confidence Change (with − without)', row=2, col=1)
        fig.update_xaxes(title_text='Confidence Change', row=2, col=2)
        fig.update_yaxes(title_text='Confidence Score', row=1, col=1)
        fig.update_yaxes(title_text='Change in Confidence', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative %', row=2, col=2)

        fig.update_layout(
            title_text='<b>Hypothesis 2: Prediction Confidence Increase Test</b>',
            height=870,
            paper_bgcolor=decision_color,
            margin=dict(t=230)
        )
        fig.add_annotation(
            text=formula_text,
            xref='paper', yref='paper',
            x=0.5, y=1.0,
            xanchor='center', yanchor='bottom',
            showarrow=False,
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='navy', borderwidth=1,
            align='left'
        )
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 3 — Environmental Sensitivity  (Pearson r per factor)
    # ══════════════════════════════════════════════════════════════════
    def plot_environmental_sensitivity(
        self,
        test_result: Dict,
        prediction_scores: np.ndarray,
        weather_params: pd.DataFrame
    ):
        """
        H3: predictions are correlated with at least one environmental factor
        FORMULA: r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / [(n−1)·σx·σy]   (Pearson r)
          σx, σy = sample std deviations of X and Y
          Also uses: Multi-factor linear regression R²
        """
        correlations = test_result.get('correlations', {})
        p_values_map = test_result.get('p_values', {})
        significant  = test_result.get('significant_factors', [])
        r_squared    = test_result.get('r_squared', float('nan'))
        reject       = test_result.get('reject_null', False)
        n            = len(prediction_scores)

        factor_names = list(correlations.keys())
        corr_values  = [correlations[f] for f in factor_names]
        p_vals_list  = [p_values_map.get(f, 1.0) for f in factor_names]
        n_sig        = len(significant)

        decision_label = f"✅ REJECT H₀  ({n_sig} factors significant)" if reject else "❌ FAIL TO REJECT H₀  (no significant correlations)"
        decision_color = "rgba(0,180,0,0.12)" if reject else "rgba(220,0,0,0.10)"

        # top factor
        top_factor_info = significant[0] if significant else (
            {'factor': factor_names[0] if factor_names else '', 'correlation': corr_values[0] if corr_values else 0}
        )
        top_f = top_factor_info.get('factor', '—')
        top_r = top_factor_info.get('correlation', float('nan'))
        top_p = p_values_map.get(top_f, float('nan'))
        top_strength = top_factor_info.get('strength', '—')

        formula_text = (
            "<b>FORMULA (Pearson Correlation per factor):</b>  "
            "r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / [(n−1)·σx·σy]<br>"
            f"  n = {n}  |  Factors tested = {len(factor_names)}  |  "
            f"Significant (p < 0.05) = {n_sig}  |  "
            f"Multi-factor regression R² = {r_squared:.4f}<br>"
            f"  Top factor: <b>{top_f}</b>  r = {top_r:.4f}  "
            f"p = {_fmt_p(top_p)}  ({top_strength})<br>"
            f"  <b>{decision_label}</b>"
        )

        specs = [
            [{'type': 'heatmap'}, {'type': 'scatter'}],
            [{'type': 'bar'},     {'type': 'bar'}]
        ]
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pearson r  Heatmap  (r = Cov/σxσy)',
                f'Top Factor Scatter: {top_f}',
                'p-values by Factor  (α = 0.05 line)',
                '|r|  Correlation Strength'
            ),
            specs=specs,
            vertical_spacing=0.20,
            horizontal_spacing=0.14
        )

        # 1 ── Heatmap
        text_ann = [[
            f"r={corr_values[i]:.3f}<br>p={_fmt_p(p_vals_list[i])}"
            for i in range(len(factor_names))
        ]]
        fig.add_trace(
            go.Heatmap(
                z=[corr_values], x=factor_names, y=['r'],
                colorscale='RdBu', zmid=0,
                text=text_ann, texttemplate='%{text}',
                textfont=dict(size=10),
                colorbar=dict(title='r', x=1.15),
                zmin=-1, zmax=1
            ),
            row=1, col=1
        )

        # 2 ── Top factor scatter + trend line
        factor_key = top_f.replace('weather_', '').replace('soil_', '')
        if factor_key in weather_params.columns:
            x_data = weather_params[factor_key].values[:len(prediction_scores)]
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=prediction_scores, mode='markers',
                    marker=dict(size=8, color=prediction_scores,
                                colorscale='Viridis', showscale=False),
                    showlegend=False
                ),
                row=1, col=2
            )
            # trend line
            from sklearn.linear_model import LinearRegression as _LR
            _m = _LR().fit(x_data.reshape(-1, 1), prediction_scores)
            _slope = float(_m.coef_[0])
            _intcp = float(_m.intercept_)
            xs = np.linspace(x_data.min(), x_data.max(), 100)
            fig.add_trace(
                go.Scatter(x=xs, y=_slope * xs + _intcp, mode='lines',
                           line=dict(color='red', width=3, dash='dash'),
                           showlegend=False),
                row=1, col=2
            )
            # annotate r & equation on scatter
            fig.add_annotation(
                x=0.02, y=0.95, xref='x2 domain', yref='y2 domain',
                text=(f'r = {top_r:.3f}<br>'
                      f'p = {_fmt_p(top_p)}<br>'
                      f'y = {_slope:.3f}·x + {_intcp:.3f}'),
                showarrow=False, align='left',
                font=dict(size=11, family='monospace'),
                bgcolor='rgba(255,255,200,0.9)',
                bordercolor='black', borderwidth=1
            )
            fig.update_xaxes(title_text=top_f, row=1, col=2)
            fig.update_yaxes(title_text='Prediction Score', row=1, col=2)

        # 3 ── p-value bar chart (log scale)
        p_colors = ['seagreen' if p < 0.05 else 'crimson' for p in p_vals_list]
        fig.add_trace(
            go.Bar(
                x=factor_names, y=p_vals_list,
                marker_color=p_colors,
                text=[_fmt_p(p) for p in p_vals_list],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_hline(y=0.05, line_dash='dash', line_color='black',
                      annotation_text='α = 0.05', row=2, col=1)
        fig.update_yaxes(title_text='p-value (log scale)', type='log', row=2, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)

        # 4 ── |r| bar chart with strength labels
        abs_r  = [abs(c) for c in corr_values]
        r_cols = ['darkgreen' if v > 0.7 else 'green' if v > 0.4 else 'orange' for v in abs_r]
        r_text = [f"|r|={v:.3f}" for v in abs_r]
        fig.add_trace(
            go.Bar(
                x=factor_names, y=abs_r,
                marker_color=r_cols,
                text=r_text, textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_hline(y=0.4, line_dash='dot', line_color='green',
                      annotation_text='Moderate (0.4)', row=2, col=2)
        fig.add_hline(y=0.7, line_dash='dot', line_color='darkgreen',
                      annotation_text='Strong (0.7)', row=2, col=2)
        fig.update_yaxes(title_text='|r|  (absolute correlation)', row=2, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=2)

        fig.update_layout(
            title_text='<b>Hypothesis 3: Environmental Sensitivity Test</b>',
            height=870,
            paper_bgcolor=decision_color,
            margin=dict(t=200)
        )
        fig.add_annotation(
            text=formula_text,
            xref='paper', yref='paper',
            x=0.5, y=1.0,
            xanchor='center', yanchor='bottom',
            showarrow=False,
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='navy', borderwidth=1,
            align='left'
        )
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 4 — NDVI Sensitivity  (Linear Regression + Pearson r)
    # ══════════════════════════════════════════════════════════════════
    def plot_ndvi_sensitivity(
        self,
        prediction_scores: np.ndarray,
        ndvi_values: np.ndarray,
        test_result: Dict
    ):
        """
        H4: prediction score correlates significantly with NDVI
        FORMULAS:
          Regression: ŷ = β₀ + β₁·NDVI
          Pearson r:  r = Cov(X,Y) / (σx·σy)
          R²:         R² = SSR / SST  =  1 − SSE/SST
          F-test:     F = (SSR/1) / (SSE/(n−2))
        """
        # ── pull values ───────────────────────────────────────────────
        slope    = test_result.get('regression_slope', float('nan'))
        intercept= test_result.get('regression_intercept', float('nan'))
        r_pearson= test_result.get('pearson_r', float('nan'))
        p_pearson= test_result.get('pearson_p_value', float('nan'))
        r_spear  = test_result.get('spearman_r', float('nan'))
        p_spear  = test_result.get('spearman_p_value', float('nan'))
        r_sq     = test_result.get('r_squared', float('nan'))
        f_stat   = test_result.get('f_statistic', float('nan'))
        f_p      = test_result.get('f_test_p_value', float('nan'))
        ci_slope = test_result.get('confidence_interval_slope_95', [float('nan'), float('nan')])
        strength = test_result.get('correlation_strength', '—')
        reject   = test_result.get('reject_null', False)
        n        = len(prediction_scores)

        decision_label = "✅ REJECT H₀  (significant NDVI correlation)" if reject else "❌ FAIL TO REJECT H₀  (no significant correlation)"
        decision_color = "rgba(0,180,0,0.12)" if reject else "rgba(220,0,0,0.10)"

        formula_text = (
            "<b>FORMULA (Linear Regression + Pearson r):</b><br>"
            "  Regression: ŷ = β₀ + β₁·NDVI  →  "
            f"ŷ = {intercept:.4f} + {slope:.4f}·NDVI<br>"
            "  Pearson r: r = Cov(X,Y)/(σx·σy)    "
            "R² = SSR/SST = 1 − SSE/SST    "
            "F = (SSR/1)/(SSE/(n−2))<br>"
            f"  n = {n}  |  "
            f"β₁ (slope) = {slope:.4f}  95%CI=[{ci_slope[0]:.4f}, {ci_slope[1]:.4f}]  |  "
            f"β₀ (intercept) = {intercept:.4f}<br>"
            f"  Pearson r = {r_pearson:.4f}  p = {_fmt_p(p_pearson)}  |  "
            f"Spearman ρ = {r_spear:.4f}  p = {_fmt_p(p_spear)}<br>"
            f"  R² = {r_sq:.4f}  (explains {r_sq*100:.1f}% of variance)  |  "
            f"F = {f_stat:.4f}  F-p = {_fmt_p(f_p)}  |  Strength: {strength}<br>"
            f"  <b>{decision_label}</b>"
        )

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'NDVI vs Prediction  (Regression + 95% PI)',
                'Residuals  (ε = y − ŷ)',
                'NDVI Distribution',
                'Prediction Score Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'histogram'}]
            ],
            vertical_spacing=0.20,
            horizontal_spacing=0.12
        )

        # ── regression prediction ─────────────────────────────────────
        y_hat   = slope * ndvi_values + intercept
        resid   = prediction_scores - y_hat
        std_res = np.std(resid)
        x_line  = np.linspace(ndvi_values.min(), ndvi_values.max(), 120)
        y_line  = slope * x_line + intercept
        y_upper = y_line + 1.96 * std_res
        y_lower = y_line - 1.96 * std_res

        # 1 ── Scatter + regression line + PI band
        fig.add_trace(
            go.Scatter(
                x=ndvi_values, y=prediction_scores, mode='markers',
                marker=dict(size=7, color=prediction_scores,
                            colorscale='Greens', showscale=True,
                            colorbar=dict(title='Score', x=1.15)),
                name='Observations'
            ),
            row=1, col=1
        )
        # 95% prediction interval band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill='toself', fillcolor='rgba(255,0,0,0.10)',
                line=dict(color='rgba(255,0,0,0)'),
                name='95% Prediction Interval'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name=f'ŷ = {slope:.3f}·NDVI + {intercept:.3f}'
            ),
            row=1, col=1
        )
        # equation + R² box
        fig.add_annotation(
            x=0.02, y=0.97, xref='x1 domain', yref='y1 domain',
            text=(f'ŷ = {slope:.3f}·x + {intercept:.3f}<br>'
                  f'R² = {r_sq:.4f}<br>'
                  f'r = {r_pearson:.4f}<br>'
                  f'p = {_fmt_p(p_pearson)}<br>'
                  f'F = {f_stat:.4f}'),
            showarrow=False, align='left',
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,200,0.9)',
            bordercolor='black', borderwidth=1
        )

        # 2 ── Residual plot
        fig.add_trace(
            go.Scatter(
                x=ndvi_values, y=resid, mode='markers',
                marker=dict(size=6, color='purple', opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=2)
        # ±1σ reference lines
        fig.add_hline(y= std_res, line_dash='dot', line_color='gray',
                      annotation_text=f'+1σ={std_res:.3f}', row=1, col=2)
        fig.add_hline(y=-std_res, line_dash='dot', line_color='gray',
                      annotation_text=f'-1σ={std_res:.3f}', row=1, col=2)
        # slope CI box
        fig.add_annotation(
            x=0.98, y=0.97, xref='x2 domain', yref='y2 domain',
            text=(f"β₁={slope:.3f}<br>"
                  f"95%CI=[{ci_slope[0]:.3f},{ci_slope[1]:.3f}]<br>"
                  f"Spearman ρ={r_spear:.3f}<br>"
                  f"ρ-p={_fmt_p(p_spear)}"),
            showarrow=False, align='right',
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,200,0.9)',
            bordercolor='black', borderwidth=1
        )

        # 3 ── NDVI histogram
        ndvi_mean = float(np.mean(ndvi_values))
        ndvi_std  = float(np.std(ndvi_values))
        fig.add_trace(
            go.Histogram(x=ndvi_values, marker_color='seagreen',
                         opacity=0.75, nbinsx=30, showlegend=False),
            row=2, col=1
        )
        fig.add_vline(x=ndvi_mean, line_dash='solid', line_color='black',
                      annotation_text=f'μ={ndvi_mean:.3f}', row=2, col=1)
        fig.add_vline(x=ndvi_mean + ndvi_std, line_dash='dot', line_color='gray',
                      annotation_text=f'+σ={ndvi_mean+ndvi_std:.3f}', row=2, col=1)
        fig.add_vline(x=ndvi_mean - ndvi_std, line_dash='dot', line_color='gray',
                      annotation_text=f'-σ={ndvi_mean-ndvi_std:.3f}', row=2, col=1)

        # 4 ── Prediction score histogram
        sc_mean = float(np.mean(prediction_scores))
        sc_std  = float(np.std(prediction_scores))
        fig.add_trace(
            go.Histogram(x=prediction_scores, marker_color='steelblue',
                         opacity=0.75, nbinsx=30, showlegend=False),
            row=2, col=2
        )
        fig.add_vline(x=sc_mean, line_dash='solid', line_color='black',
                      annotation_text=f'μ={sc_mean:.3f}', row=2, col=2)
        fig.add_vline(x=sc_mean + sc_std, line_dash='dot', line_color='gray',
                      annotation_text=f'+σ={sc_mean+sc_std:.3f}', row=2, col=2)
        fig.add_vline(x=sc_mean - sc_std, line_dash='dot', line_color='gray',
                      annotation_text=f'-σ={sc_mean-sc_std:.3f}', row=2, col=2)

        # ── axis labels ───────────────────────────────────────────────
        fig.update_xaxes(title_text='NDVI', row=1, col=1)
        fig.update_xaxes(title_text='NDVI', row=1, col=2)
        fig.update_xaxes(title_text='NDVI', row=2, col=1)
        fig.update_xaxes(title_text='Prediction Score', row=2, col=2)
        fig.update_yaxes(title_text='Prediction Score', row=1, col=1)
        fig.update_yaxes(title_text='Residual  (ε = y − ŷ)', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=2)

        fig.update_layout(
            title_text='<b>Hypothesis 4: NDVI Trend Sensitivity Test</b>',
            height=870,
            paper_bgcolor=decision_color,
            margin=dict(t=250)
        )
        fig.add_annotation(
            text=formula_text,
            xref='paper', yref='paper',
            x=0.5, y=1.0,
            xanchor='center', yanchor='bottom',
            showarrow=False,
            font=dict(size=11, family='monospace'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='navy', borderwidth=1,
            align='left'
        )
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY DASHBOARD — all 4 tests at a glance
    # ══════════════════════════════════════════════════════════════════
    def create_summary_dashboard(self, all_results: Dict):
        """
        Horizontal bar chart of p-values for all 4 tests with:
          • Formula label in bar text
          • REJECT / FAIL-TO-REJECT colour coding
          • α = 0.05 threshold line
        """
        test_mapping = {
            'fusion_significance':      ('H1: Fusion',         't = d̄/(s_d/√n)',           'p_value'),
            'confidence_increase':      ('H2: Confidence ↑',   't = d̄/(s_d/√n)  [one-tail]','p_value'),
            'environmental_sensitivity':('H3: Env Sensitivity', 'r = Cov/σxσy',              'pearson_p_value'),
            'ndvi_sensitivity':         ('H4: NDVI Trend',      'ŷ = β₀+β₁·NDVI, R²=SSR/SST','pearson_p_value'),
        }

        labels, p_vals, rejected, formulas = [], [], [], []
        for key, (name, formula, p_key) in test_mapping.items():
            if key not in all_results:
                continue
            res = all_results[key]
            # try both p-key names
            p = res.get(p_key, res.get('p_value', res.get('pearson_p_value', 1.0)))
            labels.append(name)
            p_vals.append(p)
            rejected.append(res.get('reject_null', False))
            formulas.append(formula)

        colors     = ['seagreen' if r else 'crimson' for r in rejected]
        bar_labels = [
            f"{'✅ H₀ rejected' if r else '❌ H₀ not rejected'}  |  {formula}  |  p={_fmt_p(p)}"
            for r, formula, p in zip(rejected, formulas, p_vals)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=p_vals, y=labels,
                orientation='h',
                marker_color=colors,
                text=bar_labels,
                textposition='outside',
                textfont=dict(size=11)
            )
        )
        fig.add_vline(x=0.05, line_dash='dash', line_color='black',
                      annotation_text='α = 0.05', annotation_position='top right')

        n_rejected = sum(rejected)
        validity = all_results.get('overall_summary', {}).get('model_validity', 'Unknown')
        summary_text = (
            f"<b>Hypothesis Testing Summary</b><br>"
            f"{n_rejected}/{len(labels)} hypotheses reject H₀ (p < α = 0.05)<br>"
            f"Model validity: {validity}"
        )
        fig.add_annotation(
            text=summary_text,
            xref='paper', yref='paper',
            x=0.5, y=1.18,
            xanchor='center', yanchor='top',
            showarrow=False,
            font=dict(size=14),
            bgcolor='lightgreen' if n_rejected >= 3 else 'lightyellow',
            bordercolor='black', borderwidth=2
        )

        fig.update_xaxes(title_text='p-value  (log scale)', type='log')
        fig.update_yaxes(title_text='Hypothesis Test')
        fig.update_layout(
            title_text='<b>All Hypothesis Tests — p-value Summary</b>',
            height=420,
            margin=dict(t=130, r=400),
            showlegend=False
        )
        return fig
