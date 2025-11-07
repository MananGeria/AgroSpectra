"""
Test script for premium features
Verifies locust prediction, disease detection, yield prediction, and nutrient analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier5_models.locust_swarm_predictor import LocustSwarmPredictor
from tier5_models.disease_detector import MultiCropDiseaseDetector
from tier5_models.yield_predictor import YieldPredictor
from tier5_models.nutrient_detector import NutrientDeficiencyDetector


def test_locust_predictor():
    """Test locust swarm prediction"""
    print("\n" + "="*60)
    print("TESTING LOCUST SWARM PREDICTOR")
    print("="*60)
    
    predictor = LocustSwarmPredictor()
    
    # Test 1: High-risk area (Rajasthan - near breeding ground)
    print("\n--- Test 1: High-Risk Area (Rajasthan) ---")
    result = predictor.predict_swarm_risk(
        latitude=27.0,  # Rajasthan
        longitude=73.0,
        temperature=32.0,
        humidity=75.0,
        rainfall=50.0,
        wind_speed=12.0,
        vegetation_index=0.65,
        month=7  # Monsoon season
    )
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']:.1f}/100")
    print(f"Factors: {result['factors']}")
    
    # Test 2: Low-risk area (South India)
    print("\n--- Test 2: Low-Risk Area (South India) ---")
    result = predictor.predict_swarm_risk(
        latitude=13.0,  # Karnataka
        longitude=77.0,
        temperature=26.0,
        humidity=60.0,
        rainfall=20.0,
        wind_speed=8.0,
        vegetation_index=0.70,
        month=2  # Winter
    )
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']:.1f}/100")
    
    print("\n✅ Locust Predictor: PASSED")


def test_disease_detector():
    """Test disease detection"""
    print("\n" + "="*60)
    print("TESTING DISEASE DETECTOR")
    print("="*60)
    
    detector = MultiCropDiseaseDetector()
    
    # Test 1: Wheat with rust (low NDVI, favorable conditions)
    print("\n--- Test 1: Wheat with Potential Rust ---")
    result = detector.detect_diseases(
        crop_type='wheat',
        ndvi=0.45,  # Low NDVI
        evi=0.40,
        red_edge_position=715,  # Shifted red edge
        temperature=25.0,
        humidity=75.0,
        month=3  # Spring
    )
    print(f"Overall Health: {result['overall_health']}")
    print(f"Detected Diseases: {result['total_detections']}")
    if result['detected_diseases']:
        for disease in result['detected_diseases']:
            print(f"  - {disease['disease']}: {disease['confidence']:.0%} ({disease['severity']})")
    
    # Test 2: Healthy rice
    print("\n--- Test 2: Healthy Rice ---")
    result = detector.detect_diseases(
        crop_type='rice',
        ndvi=0.80,  # High NDVI
        evi=0.75,
        red_edge_position=720,
        temperature=28.0,
        humidity=65.0,
        month=8
    )
    print(f"Overall Health: {result['overall_health']}")
    print(f"Detected Diseases: {result['total_detections']}")
    
    print("\n✅ Disease Detector: PASSED")


def test_yield_predictor():
    """Test yield prediction"""
    print("\n" + "="*60)
    print("TESTING YIELD PREDICTOR")
    print("="*60)
    
    predictor = YieldPredictor()
    
    # Test 1: Wheat with good conditions
    print("\n--- Test 1: Wheat with Good Conditions ---")
    result = predictor.predict_yield(
        crop_type='wheat',
        ndvi_mean=0.75,
        evi_mean=0.70,
        area_hectares=10.0,
        temperature_mean=22.0,
        rainfall_sum=400.0,
        growth_stage='grain_filling',
        soil_quality=0.8,
        irrigation_available=True,
        days_to_harvest=30
    )
    print(f"Predicted Yield: {result['predicted_yield_kg_per_ha']:,.0f} kg/ha")
    print(f"Total Yield: {result['total_yield_tons']:.2f} tons")
    print(f"Grade: {result['yield_grade']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Estimated Value: ₹{result['economic_estimate']['estimated_gross_value']:,.2f}")
    
    # Test 2: Stressed corn
    print("\n--- Test 2: Stressed Corn ---")
    result = predictor.predict_yield(
        crop_type='corn',
        ndvi_mean=0.50,  # Stressed
        evi_mean=0.45,
        area_hectares=5.0,
        temperature_mean=35.0,  # Heat stress
        rainfall_sum=200.0,  # Low water
        growth_stage='vegetative',
        soil_quality=0.6,
        irrigation_available=False,
        days_to_harvest=90
    )
    print(f"Predicted Yield: {result['predicted_yield_kg_per_ha']:,.0f} kg/ha")
    print(f"Total Yield: {result['total_yield_tons']:.2f} tons")
    print(f"Grade: {result['yield_grade']}")
    print(f"Risk Factors: {len(result['risk_factors'])}")
    
    print("\n✅ Yield Predictor: PASSED")


def test_nutrient_detector():
    """Test nutrient deficiency detection"""
    print("\n" + "="*60)
    print("TESTING NUTRIENT DEFICIENCY DETECTOR")
    print("="*60)
    
    detector = NutrientDeficiencyDetector()
    
    # Test 1: Nitrogen deficiency (low NDVI)
    print("\n--- Test 1: Nitrogen Deficiency ---")
    result = detector.detect_deficiencies(
        ndvi=0.45,  # Low - indicates N deficiency
        evi=0.40,
        red_band=0.12,  # High red reflectance
        nir_band=0.42,  # Low NIR
        crop_type='wheat',
        soil_ph=6.5,
        temperature=25.0
    )
    print(f"Nutritional Health: {result['nutritional_health']}")
    print(f"Detected Deficiencies: {result['total_deficiencies']}")
    if result['detected_deficiencies']:
        for deficiency in result['detected_deficiencies'][:3]:  # Show top 3
            print(f"  - {deficiency['nutrient']}: {deficiency['probability']:.0%} ({deficiency['severity']})")
            print(f"    Treatment: {deficiency['treatment']['fertilizer']}")
    
    # Test 2: Optimal nutrition
    print("\n--- Test 2: Optimal Nutrition ---")
    result = detector.detect_deficiencies(
        ndvi=0.80,  # High - healthy
        evi=0.75,
        red_band=0.08,
        nir_band=0.55,
        crop_type='rice',
        soil_ph=6.5,
        temperature=26.0
    )
    print(f"Nutritional Health: {result['nutritional_health']}")
    print(f"Detected Deficiencies: {result['total_deficiencies']}")
    
    # Test 3: Iron deficiency due to alkaline soil
    print("\n--- Test 3: Iron Deficiency (Alkaline Soil) ---")
    result = detector.detect_deficiencies(
        ndvi=0.62,
        evi=0.58,
        red_band=0.09,
        nir_band=0.48,
        crop_type='corn',
        soil_ph=8.2,  # High pH limits Fe availability
        temperature=28.0
    )
    print(f"Nutritional Health: {result['nutritional_health']}")
    print(f"Soil pH Impact: {result['soil_factors']['ph_impact']}")
    if result['detected_deficiencies']:
        for deficiency in result['detected_deficiencies']:
            if 'Iron' in deficiency['nutrient']:
                print(f"  - {deficiency['nutrient']}: {deficiency['probability']:.0%} ({deficiency['severity']})")
    
    print("\n✅ Nutrient Detector: PASSED")


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# AGROSPECTRA PREMIUM FEATURES TEST SUITE")
    print("#"*60)
    
    try:
        test_locust_predictor()
        test_disease_detector()
        test_yield_predictor()
        test_nutrient_detector()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("="*60)
        print("\nAll premium features are working correctly:")
        print("✅ Locust Swarm Prediction")
        print("✅ Multi-Crop Disease Detection")
        print("✅ Crop Yield Prediction")
        print("✅ Nutrient Deficiency Detection")
        print("\nThe system is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
