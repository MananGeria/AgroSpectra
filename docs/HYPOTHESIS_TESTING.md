# 🔬 Hypothesis Testing Framework for AgroSpectra

## Overview

This document describes the statistical hypothesis testing framework implemented in AgroSpectra to validate the machine learning models and data fusion approach used for agricultural monitoring.

## Purpose

The hypothesis testing framework serves to:
- **Validate model performance** through rigorous statistical analysis
- **Quantify the impact** of data fusion on prediction quality
- **Ensure environmental sensitivity** of the prediction models
- **Verify NDVI-prediction relationships** for vegetation health monitoring

## The Four Critical Hypotheses

### 1. Fusion Significance Hypothesis 🔀

**Research Question:** Does fusing multiple data sources (satellite, weather, soil) significantly improve prediction accuracy compared to using individual sources?

#### Hypotheses
- **H0 (Null):** Fusion does not significantly change prediction score
- **H1 (Alternative):** Fusion significantly changes prediction score

#### Statistical Test
- **Method:** Paired t-test
- **Rationale:** Compares the same samples before and after fusion, controlling for sample-specific variation

#### Metrics Evaluated
- **t-statistic:** Magnitude of difference relative to variability
- **p-value:** Probability of observing such differences by chance
- **Cohen's d:** Effect size (magnitude of practical significance)
  - Small: 0.2 - 0.5
  - Medium: 0.5 - 0.8
  - Large: > 0.8
- **Confidence Interval:** Range of plausible mean differences

#### Interpretation
- **Reject H0 if p < 0.05:** Fusion produces statistically significant changes
- **Cohen's d indicates practical importance:** Large effect size means fusion has meaningful impact

#### Example Result
```
p-value: 0.0023
Cohen's d: 0.76 (Medium effect)
Mean difference: +0.085
Conclusion: Fusion SIGNIFICANTLY improves predictions
```

---

### 2. Prediction Confidence Increase Hypothesis 📈

**Research Question:** Does data fusion increase the model's confidence in its predictions?

#### Hypotheses
- **H0 (Null):** Fusion does not increase model confidence
- **H1 (Alternative):** Fusion increases prediction confidence level

#### Statistical Test
- **Method:** One-sided paired t-test (greater)
- **Rationale:** Tests specifically for confidence increase (directional hypothesis)

#### Metrics Evaluated
- **t-statistic:** One-sided test for increase
- **p-value:** Probability that observed increase is due to chance
- **Mean increase:** Average confidence gain
- **Improvement rate:** Percentage of samples showing confidence increase
- **Effect size:** Cohen's d for magnitude of improvement

#### Interpretation
- **Reject H0 if p < 0.05:** Fusion significantly increases confidence
- **Improvement rate:** Percentage of predictions that became more confident

#### Example Result
```
p-value: 0.0001
Mean increase: +0.125 (12.5%)
Improvement rate: 87.3% of samples improved
Conclusion: Fusion SIGNIFICANTLY increases model confidence
```

---

### 3. Environmental Sensitivity Hypothesis 🌡️

**Research Question:** Are crop health predictions appropriately sensitive to environmental factors (weather, soil conditions)?

#### Hypotheses
- **H0 (Null):** Weather/soil parameters do not significantly influence prediction output
- **H1 (Alternative):** Weather/soil parameters significantly influence prediction output

#### Statistical Test
- **Method:** Multiple Pearson correlation analysis
- **Rationale:** Quantifies linear relationships between environmental factors and predictions

#### Metrics Evaluated
- **Correlation coefficients (r):** Strength and direction of relationships
  - Weak: |r| < 0.3
  - Moderate: 0.3 ≤ |r| < 0.7
  - Strong: |r| ≥ 0.7
- **p-values:** Significance of each correlation
- **R² (Multiple regression):** Variance in predictions explained by environmental factors
- **Significant factors:** Environmental variables with p < 0.05

#### Interpretation
- **Reject H0 if ANY factor is significant:** Model responds to environmental conditions
- **R² indicates overall predictive power:** Higher values mean environment explains more variance

#### Example Result
```
Significant factors found: 3
- Temperature: r=0.68, p<0.001 (Moderate correlation)
- Humidity: r=0.54, p=0.002 (Moderate correlation)
- Rainfall: r=0.42, p=0.015 (Moderate correlation)

R²=0.52 (Environmental factors explain 52% of prediction variance)
Conclusion: Environmental parameters SIGNIFICANTLY influence predictions
```

---

### 4. NDVI Trend Sensitivity Hypothesis 🌿

**Research Question:** Do crop health predictions respond appropriately to changes in vegetation index (NDVI)?

#### Hypotheses
- **H0 (Null):** Prediction score does not significantly change with NDVI variation
- **H1 (Alternative):** Prediction score significantly changes with NDVI variation

#### Statistical Test
- **Methods:**
  - Pearson correlation (linear relationship)
  - Spearman correlation (monotonic relationship)
  - Linear regression (predictive relationship)
  - F-test (significance of regression model)

#### Metrics Evaluated
- **Pearson r:** Linear correlation strength
- **Spearman ρ:** Non-parametric correlation
- **R²:** Variance explained by NDVI
- **Regression slope:** Rate of change (prediction per unit NDVI)
- **F-statistic:** Overall model significance

#### Interpretation
- **Reject H0 if p < 0.05:** Predictions respond to NDVI changes
- **Positive slope expected:** Higher NDVI → Better health predictions
- **R² indicates NDVI's importance:** How much NDVI explains prediction variance

#### Example Result
```
Pearson r: 0.82, p<0.0001 (Strong correlation)
R²: 0.67 (NDVI explains 67% of prediction variance)
Regression: y = 0.745x + 0.123
Slope: For every 0.1 increase in NDVI, prediction increases by 0.075

Conclusion: Prediction SIGNIFICANTLY changes with NDVI
Model appropriately responds to vegetation health indicators
```

---

## Overall Model Validation

After running all four tests, an overall model validity assessment is provided:

### Validity Levels
- **High Validity:** 3-4 tests reject H0
  - Model demonstrates strong statistical validation
  - All key relationships are significant
  - Safe for production deployment

- **Moderate Validity:** 2 tests reject H0
  - Model shows acceptable validation
  - Some relationships need strengthening
  - Suitable for use with monitoring

- **Low Validity:** 0-1 tests reject H0
  - Model requires significant improvement
  - Lacks statistical evidence for effectiveness
  - Needs retraining or architecture changes

### Example Summary
```
Total Tests: 4
Significant Results: 4
Significance Rate: 100%
Model Validity: HIGH

✅ All hypotheses validated - Model ready for production!
```

---

## Implementation Details

### File Structure
```
src/tier5_models/
├── hypothesis_testing.py          # Core testing logic
└── hypothesis_visualizations.py   # Visualization utilities

scripts/
└── run_hypothesis_tests.py        # Standalone test runner

results/
├── hypothesis_test_results.csv    # Test summary
└── hypothesis_visualizations/     # Interactive HTML plots
```

### Running Tests

#### Option 1: Standalone Script
```bash
python scripts/run_hypothesis_tests.py
```

#### Option 2: From Dashboard
Navigate to the "🔬 Hypothesis Testing" tab in the AgroSpectra dashboard after running an analysis.

#### Option 3: Programmatically
```python
from tier5_models.hypothesis_testing import HypothesisTester, generate_synthetic_test_data

# Generate or load test data
data = generate_synthetic_test_data(n_samples=100)

# Run tests
tester = HypothesisTester()
results = tester.run_all_tests(**data)

# Generate report
report = tester.generate_report('output.csv')
```

---

## Visualizations

Each hypothesis test includes comprehensive visualizations:

### Test 1: Fusion Significance
- Distribution comparison (before vs after)
- Scatter plot (paired comparison)
- Difference distribution
- Time series comparison

### Test 2: Confidence Increase
- Box plots (confidence distributions)
- Individual sample changes
- Improvement distribution histogram
- Cumulative improvement curve

### Test 3: Environmental Sensitivity
- Correlation heatmap
- Scatter plot with trend line (top factor)
- P-value significance chart
- Factor contribution analysis

### Test 4: NDVI Sensitivity
- Scatter plot with regression line and confidence interval
- Residual plot
- NDVI distribution
- Prediction distribution

### Summary Dashboard
- P-value comparison across all tests
- Significance threshold visualization
- Overall validity assessment

---

## Statistical Significance Criteria

### Significance Level (α)
- **Default α = 0.05** (5% significance level)
- Means 95% confidence in rejecting H0
- Industry standard for agricultural research

### P-value Interpretation
- **p < 0.001:** Extremely strong evidence against H0
- **p < 0.01:** Very strong evidence against H0
- **p < 0.05:** Strong evidence against H0 (reject H0)
- **p ≥ 0.05:** Insufficient evidence against H0 (fail to reject H0)

### Effect Size Guidelines
- **Cohen's d:**
  - Negligible: < 0.2
  - Small: 0.2 - 0.5
  - Medium: 0.5 - 0.8
  - Large: > 0.8

- **Correlation (r):**
  - Weak: |r| < 0.3
  - Moderate: 0.3 ≤ |r| < 0.7
  - Strong: |r| ≥ 0.7

---

## Best Practices

### Data Requirements
- **Minimum sample size:** 30 observations per test
- **Recommended:** 100+ observations for robust results
- **Paired data:** Same samples before/after fusion for Tests 1-2
- **Independent observations:** No temporal autocorrelation

### Model Integration
1. **Baseline tracking:** Store predictions before fusion
2. **Confidence scoring:** Implement confidence estimation
3. **Environmental logging:** Record all environmental inputs
4. **NDVI time series:** Maintain historical NDVI data

### Validation Workflow
1. **Initial validation:** Run tests with training data
2. **Production monitoring:** Periodic revalidation (monthly)
3. **Model updates:** Retest after any model changes
4. **Regional validation:** Test across different geographical regions

---

## References

### Statistical Methods
- Paired t-test: Student's t-test for dependent samples
- Pearson correlation: Measures linear relationships
- Spearman correlation: Non-parametric rank correlation
- Linear regression: Ordinary least squares (OLS)
- F-test: Analysis of variance for regression

### Agricultural Context
- NDVI as vegetation health indicator (Tucker, 1979)
- Multi-source data fusion in precision agriculture (Zhang & Kovacs, 2012)
- Environmental factors in crop modeling (Jones et al., 2017)

### Software Libraries
- **scipy.stats:** Statistical testing and distributions
- **sklearn:** Machine learning and regression
- **plotly:** Interactive visualizations
- **pandas/numpy:** Data manipulation

---

## Troubleshooting

### Common Issues

**Issue:** Tests show no significance (all H0 accepted)
- **Cause:** Model not learning, poor data quality, or insufficient fusion
- **Solution:** Check model architecture, verify data quality, ensure proper fusion implementation

**Issue:** Conflicting results (some tests pass, others fail)
- **Cause:** Partial model effectiveness or dataset-specific issues
- **Solution:** Investigate failed tests, analyze data distribution, consider feature engineering

**Issue:** Very low p-values (p < 0.0001) for all tests
- **Cause:** Very large effect sizes or large sample sizes
- **Solution:** Verify results are not due to data leakage or overfitting

### Getting Help
- Check logs in `hypothesis_testing.log`
- Review visualization outputs for anomalies
- Consult detailed error messages in dashboard
- Verify input data formats and ranges

---

## Future Enhancements

1. **Temporal validation:** Test prediction stability over time
2. **Cross-regional validation:** Compare performance across regions
3. **Multi-crop analysis:** Separate validation per crop type
4. **Robustness testing:** Performance under various conditions
5. **Bayesian methods:** Alternative statistical frameworks

---

## Conclusion

The hypothesis testing framework provides rigorous statistical validation of the AgroSpectra platform. By testing fusion effectiveness, confidence improvements, environmental sensitivity, and NDVI relationships, we ensure that predictions are scientifically sound and practically useful for agricultural decision-making.

Regular validation using this framework maintains model quality and builds trust with users through transparent, evidence-based performance metrics.

---

*Last Updated: February 26, 2026*
*Version: 1.0*
*Author: AgroSpectra Development Team*
