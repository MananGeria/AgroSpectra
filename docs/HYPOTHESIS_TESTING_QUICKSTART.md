# 🚀 Quick Start Guide - Hypothesis Testing

## Run Tests in 3 Steps

### 1. Standalone Mode (Recommended for Testing)

```bash
# Navigate to project directory
cd /Users/shubhjyot/AgroSpectra

# Run hypothesis tests
python scripts/run_hypothesis_tests.py
```

**Output:**
- CSV report: `results/hypothesis_test_results.csv`
- Interactive visualizations: `results/hypothesis_visualizations/*.html`

### 2. Dashboard Integration

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py
```

1. Complete a crop analysis
2. Navigate to "🔬 Hypothesis Testing" tab
3. View interactive results

### 3. Programmatic Usage

```python
from tier5_models.hypothesis_testing import HypothesisTester, generate_synthetic_test_data

# Generate test data
data = generate_synthetic_test_data(n_samples=100)

# Initialize and run tests
tester = HypothesisTester()
results = tester.run_all_tests(**data)

# Print summary
print(f"Model Validity: {results['overall_summary']['model_validity']}")
```

---

## Quick Reference Table

| Hypothesis | Test Type | Key Metric | Reject H0 if | Interpretation |
|------------|-----------|------------|--------------|----------------|
| **1. Fusion Significance** | Paired t-test | p-value | p < 0.05 | Fusion changes predictions significantly |
| **2. Confidence Increase** | One-sided t-test | p-value | p < 0.05 | Fusion increases model confidence |
| **3. Environmental Sensitivity** | Correlation | Any factor p < 0.05 | Yes | Model responds to environment |
| **4. NDVI Sensitivity** | Regression | p-value | p < 0.05 | Model responds to vegetation health |

---

## Interpreting Results

### ✅ **High Validity** (3-4 tests pass)
- Model is statistically validated
- Ready for production use
- All key relationships confirmed

### ⚠️ **Moderate Validity** (2 tests pass)
- Acceptable validation
- Monitor performance
- Consider improvements

### ❌ **Low Validity** (0-1 tests pass)
- Model needs improvement
- Retrain or redesign required
- Not recommended for production

---

## Example Output

```
HYPOTHESIS TEST RESULTS
==============================================
Test                          p-value  Result
----------------------------------------------
Fusion Significance           0.0023   ✅ Significant
Confidence Increase           0.0001   ✅ Significant
Environmental Sensitivity     0.0156   ✅ Significant
NDVI Sensitivity             0.0000   ✅ Significant
----------------------------------------------

Model Validity: HIGH (4/4 tests passed)
✅ Model ready for production!
```

---

## Common Commands

```bash
# Create results directory
mkdir -p results/hypothesis_visualizations

# Run tests with logging
python scripts/run_hypothesis_tests.py > test_output.txt 2>&1

# View results
open results/hypothesis_visualizations/summary_dashboard.html

# Check logs
tail -f hypothesis_testing.log
```

---

## Dependencies

```bash
# Required packages
pip install scipy scikit-learn plotly pandas numpy loguru
```

All dependencies are included in `requirements.txt`.

---

## Troubleshooting

**Error: Module not found**
```bash
# Ensure you're in the correct directory
cd /Users/shubhjyot/AgroSpectra
python -m pip install -r requirements.txt
```

**Error: No test data**
```bash
# The script generates synthetic data automatically
# Check that src/tier5_models/hypothesis_testing.py exists
```

**Low p-values (all < 0.0001)**
- This is expected with strong effects
- Verify data quality
- Check for data leakage

---

## Quick Tips

💡 **Tip 1:** Run tests monthly to track model performance over time

💡 **Tip 2:** Save results with timestamps for comparison
```bash
python scripts/run_hypothesis_tests.py
mv results/hypothesis_test_results.csv results/results_$(date +%Y%m%d).csv
```

💡 **Tip 3:** Use visualizations in presentations
- All plots are interactive HTML files
- Embed in reports or dashboards

💡 **Tip 4:** Compare before/after model changes
- Run tests before changes (baseline)
- Run tests after changes
- Compare validity scores

---

## Next Steps

1. ✅ Run tests to establish baseline
2. 📊 Review visualizations
3. 📈 Monitor over time
4. 🔄 Revalidate after model updates
5. 📄 Share results with stakeholders

---

For detailed documentation, see [HYPOTHESIS_TESTING.md](HYPOTHESIS_TESTING.md)
