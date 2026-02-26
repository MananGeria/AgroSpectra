"""
Standalone Hypothesis Testing Demo Script
Demonstrates all four hypothesis tests with visualizations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier5_models.hypothesis_testing import HypothesisTester, generate_synthetic_test_data
from tier5_models.hypothesis_visualizations import HypothesisVisualizer
from loguru import logger

# Configure logger
logger.add("hypothesis_testing.log", rotation="10 MB")


def main():
    """Run comprehensive hypothesis testing demonstration"""
    
    print("="*80)
    print(" 🔬 AGROSPECTRA HYPOTHESIS TESTING FRAMEWORK")
    print("="*80)
    print()
    
    # Generate test data
    print("📊 Generating synthetic test data (n=150 samples)...")
    test_data = generate_synthetic_test_data(n_samples=150)
    print("✅ Test data generated successfully\n")
    
    # Initialize tester and visualizer
    tester = HypothesisTester()
    visualizer = HypothesisVisualizer()
    
    print("="*80)
    print(" Running Statistical Hypothesis Tests")
    print("="*80)
    print()
    
    # Run all tests
    results = tester.run_all_tests(**test_data)
    
    # Print results
    print("\n" + "="*80)
    print(" TEST RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Generate and display report
    report = tester.generate_report('results/hypothesis_test_results.csv')
    print(report.to_string(index=False))
    print()
    
    # Print detailed results
    for test_name, test_result in results.items():
        if test_name != 'overall_summary':
            print("\n" + "="*80)
            print(f" {test_result['hypothesis'].upper()}")
            print("="*80)
            print()
            print(f"H0: {test_result['h0']}")
            print(f"H1: {test_result['h1']}")
            print(f"Test Type: {test_result['test_type']}")
            print()
            
            # Get p-value (handle different field names)
            p_val = test_result.get('p_value', test_result.get('pearson_p_value', 'N/A'))
            if isinstance(p_val, (int, float)):
                print(f"p-value: {p_val:.6f}")
                print(f"Significance Level (α): {test_result['alpha']}")
                print(f"Decision: {'REJECT H0' if test_result['reject_null'] else 'FAIL TO REJECT H0'}")
            
            print()
            print("CONCLUSION:")
            print(f"  {test_result['conclusion']}")
            print()
            print("INTERPRETATION:")
            print(f"  {test_result['interpretation']}")
            print()
    
    # Overall summary
    overall = results.get('overall_summary', {})
    print("\n" + "="*80)
    print(" OVERALL MODEL VALIDATION")
    print("="*80)
    print()
    print(f"Total Tests: {overall.get('total_tests', 0)}")
    print(f"Significant Results: {overall.get('significant_results', 0)}")
    print(f"Significance Rate: {overall.get('significance_rate', 0):.1%}")
    print(f"Model Validity: {overall.get('model_validity', 'Unknown')}")
    print()
    
    # Generate visualizations
    print("="*80)
    print(" Generating Visualizations")
    print("="*80)
    print()
    
    # Save visualizations
    output_dir = Path('results/hypothesis_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Creating visualization 1/5: Fusion Significance...")
    fig1 = visualizer.plot_fusion_comparison(
        test_data['predictions_without_fusion'],
        test_data['predictions_with_fusion'],
        results['fusion_significance']
    )
    fig1.write_html(str(output_dir / 'test1_fusion_significance.html'))
    print(f"   ✅ Saved to {output_dir / 'test1_fusion_significance.html'}")
    
    print("📊 Creating visualization 2/5: Confidence Increase...")
    fig2 = visualizer.plot_confidence_increase(
        test_data['confidence_without_fusion'],
        test_data['confidence_with_fusion'],
        results['confidence_increase']
    )
    fig2.write_html(str(output_dir / 'test2_confidence_increase.html'))
    print(f"   ✅ Saved to {output_dir / 'test2_confidence_increase.html'}")
    
    print("📊 Creating visualization 3/5: Environmental Sensitivity...")
    fig3 = visualizer.plot_environmental_sensitivity(
        results['environmental_sensitivity'],
        test_data['prediction_scores'],
        test_data['weather_params']
    )
    fig3.write_html(str(output_dir / 'test3_environmental_sensitivity.html'))
    print(f"   ✅ Saved to {output_dir / 'test3_environmental_sensitivity.html'}")
    
    print("📊 Creating visualization 4/5: NDVI Sensitivity...")
    fig4 = visualizer.plot_ndvi_sensitivity(
        test_data['prediction_scores'],
        test_data['ndvi_values'],
        results['ndvi_sensitivity']
    )
    fig4.write_html(str(output_dir / 'test4_ndvi_sensitivity.html'))
    print(f"   ✅ Saved to {output_dir / 'test4_ndvi_sensitivity.html'}")
    
    print("📊 Creating visualization 5/5: Summary Dashboard...")
    fig5 = visualizer.create_summary_dashboard(results)
    fig5.write_html(str(output_dir / 'summary_dashboard.html'))
    print(f"   ✅ Saved to {output_dir / 'summary_dashboard.html'}")
    
    print()
    print("="*80)
    print(" ✅ HYPOTHESIS TESTING COMPLETE")
    print("="*80)
    print()
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print(f"📄 CSV Report: results/hypothesis_test_results.csv")
    print()
    print("🌐 Open the HTML files in your browser to view interactive visualizations!")
    print()
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Hypothesis testing completed successfully")
    except Exception as e:
        logger.error(f"Error during hypothesis testing: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
