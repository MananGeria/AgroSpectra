"""Quick smoke test for new hypothesis_visualizations.py"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')

from tier5_models.hypothesis_testing import HypothesisTester, generate_synthetic_test_data
from tier5_models.hypothesis_visualizations import HypothesisVisualizer, _fmt_p

# _fmt_p helper
assert _fmt_p(1e-10) == "1.00e-10", f"got {_fmt_p(1e-10)}"
assert _fmt_p(0.032) == "0.0320",   f"got {_fmt_p(0.032)}"
print("_fmt_p OK")

data  = generate_synthetic_test_data(n_samples=100, scenario='realistic')
tstr  = HypothesisTester()

wp = data.get('weather_params', {})
env_df = pd.DataFrame(wp) if wp else pd.DataFrame({
    'temperature': np.random.rand(100),
    'rainfall':    np.random.rand(100)
})

all_r = tstr.run_all_tests(
    predictions_without_fusion=data['predictions_without'],
    predictions_with_fusion=data['predictions_with'],
    confidence_without_fusion=data['confidence_without'],
    confidence_with_fusion=data['confidence_with'],
    prediction_scores=data['predictions_without'],
    ndvi_values=data['ndvi_values'],
    weather_params=env_df
)
viz   = HypothesisVisualizer()

# H1
fig1 = viz.plot_fusion_comparison(
    data['predictions_without'], data['predictions_with'],
    all_r['fusion_significance']
)
print("H1 OK  | reject =", all_r['fusion_significance']['reject_null'],
      "| p =", _fmt_p(all_r['fusion_significance']['p_value']))

# H2
fig2 = viz.plot_confidence_increase(
    data['confidence_without'], data['confidence_with'],
    all_r['confidence_increase']
)
print("H2 OK  | reject =", all_r['confidence_increase']['reject_null'],
      "| p =", _fmt_p(all_r['confidence_increase']['p_value']))

# H3 — env_df already built above
fig3 = viz.plot_environmental_sensitivity(
    all_r['environmental_sensitivity'],
    data['predictions_without'],
    env_df
)
r3 = all_r['environmental_sensitivity']
print("H3 OK  | reject =", r3['reject_null'],
      "| sig factors =", len(r3.get('significant_factors', [])))

# H4
fig4 = viz.plot_ndvi_sensitivity(
    data['predictions_without'],
    data['ndvi_values'],
    all_r['ndvi_sensitivity']
)
r4 = all_r['ndvi_sensitivity']
print("H4 OK  | reject =", r4['reject_null'],
      "| r =", round(r4['pearson_r'], 4),
      "| R2 =", round(r4['r_squared'], 4),
      "| p =", _fmt_p(r4['pearson_p_value']))

# Summary
fig_s = viz.create_summary_dashboard(all_r)
print("Summary OK")

print("\nAll checks passed!")
