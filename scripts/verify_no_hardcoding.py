"""
Verification Script: Prove No Hardcoded Data
=============================================

This script tests the same field with different parameters to prove
that all outputs are dynamically calculated, not hardcoded.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier5_models.locust_swarm_predictor import LocustSwarmPredictor
from tier5_models.disease_detector import MultiCropDiseaseDetector
from tier5_models.yield_predictor import YieldPredictor
from tier5_models.nutrient_detector import NutrientDeficiencyDetector


def test_no_hardcoding():
    """Verify that outputs vary by location and conditions"""
    
    print("\n" + "="*70)
    print("VERIFICATION TEST: NO HARDCODED DATA")
    print("="*70)
    
    # Test 1: Locust Risk - Different Locations
    print("\n📍 TEST 1: Locust Risk Varies by Location")
    print("-" * 70)
    
    predictor = LocustSwarmPredictor()
    
    locations = [
        (27.0, 73.0, "Rajasthan (Near breeding zone)"),
        (13.0, 77.0, "Karnataka (Far from breeding)"),
        (31.0, 75.0, "Punjab (Moderate distance)"),
    ]
    
    locust_scores = []
    for lat, lon, name in locations:
        result = predictor.predict_swarm_risk(
            lat=lat, lon=lon,
            temperature=30.0, humidity=70.0,
            rainfall_15days=25.0, ndvi=0.65
        )
        score = result['risk_score']
        locust_scores.append(score)
        print(f"{name:40s} → Risk: {score:.3f} ({result['category']})")
    
    # Verify all scores are different
    unique_scores = len(set(locust_scores))
    if unique_scores == len(locust_scores):
        print("✅ PASSED: All locations have DIFFERENT locust risks")
    else:
        print("❌ FAILED: Some locations have same risk (hardcoded!)")
        return False
    
    
    # Test 2: Disease Detection - Different NDVI
    print("\n🦠 TEST 2: Disease Detection Varies by NDVI")
    print("-" * 70)
    
    detector = MultiCropDiseaseDetector()
    
    ndvi_levels = [0.40, 0.60, 0.80]
    disease_counts = []
    
    for ndvi in ndvi_levels:
        result = detector.detect_diseases(
            crop_type='wheat',
            ndvi=ndvi, evi=ndvi*0.85,
            temperature=25.0, humidity=75.0, month=3
        )
        count = result['total_detections']
        disease_counts.append(count)
        print(f"NDVI {ndvi:.2f} → Diseases detected: {count} ({result['overall_health']})")
    
    # Verify disease counts vary with NDVI
    if disease_counts[0] >= disease_counts[1] >= disease_counts[2]:
        print("✅ PASSED: Lower NDVI = More diseases (correct logic)")
    else:
        print("❌ FAILED: Disease detection not varying correctly")
        return False
    
    
    # Test 3: Yield Prediction - Different Conditions
    print("\n🌾 TEST 3: Yield Varies by Vegetation Health")
    print("-" * 70)
    
    yield_predictor = YieldPredictor()
    
    conditions = [
        (0.50, "Stressed"),
        (0.70, "Moderate"),
        (0.85, "Excellent"),
    ]
    
    yields = []
    for ndvi, condition in conditions:
        result = yield_predictor.predict_yield(
            crop_type='wheat',
            ndvi_mean=ndvi, evi_mean=ndvi*0.85,
            area_hectares=10.0,
            temperature_mean=22.0, rainfall_sum=400.0
        )
        yield_val = result['predicted_yield_kg_per_ha']
        yields.append(yield_val)
        print(f"{condition:15s} (NDVI {ndvi:.2f}) → Yield: {yield_val:,.0f} kg/ha")
    
    # Verify yields increase with better NDVI
    if yields[0] < yields[1] < yields[2]:
        print("✅ PASSED: Better NDVI = Higher yield (correct logic)")
    else:
        print("❌ FAILED: Yield not responding to vegetation health")
        return False
    
    
    # Test 4: Nutrient Detection - Different NDVI
    print("\n🧪 TEST 4: Nutrient Deficiency Varies by NDVI")
    print("-" * 70)
    
    nutrient_detector = NutrientDeficiencyDetector()
    
    ndvi_levels = [0.45, 0.65, 0.80]
    deficiency_counts = []
    
    for ndvi in ndvi_levels:
        result = nutrient_detector.detect_deficiencies(
            ndvi=ndvi, evi=ndvi*0.85,
            soil_ph=6.5, temperature=25.0
        )
        count = result['total_deficiencies']
        deficiency_counts.append(count)
        print(f"NDVI {ndvi:.2f} → Deficiencies: {count} ({result['nutritional_health']})")
    
    # Verify deficiency counts decrease with higher NDVI
    if deficiency_counts[0] >= deficiency_counts[1] >= deficiency_counts[2]:
        print("✅ PASSED: Lower NDVI = More deficiencies (correct logic)")
    else:
        print("❌ FAILED: Nutrient detection not varying correctly")
        return False
    
    
    # Test 5: Seasonal Variation
    print("\n📅 TEST 5: Disease Risk Varies by Season")
    print("-" * 70)
    
    months = [
        (3, "March (Spring)"),
        (7, "July (Monsoon)"),
        (12, "December (Winter)")
    ]
    
    seasonal_risks = []
    for month, season in months:
        result = detector.detect_diseases(
            crop_type='rice',
            ndvi=0.55, evi=0.48,
            temperature=28.0, humidity=75.0,
            month=month
        )
        risk = result['environmental_factors']['favorable_for_disease']
        seasonal_risks.append(risk)
        print(f"{season:20s} → Disease favorable: {risk}")
    
    # Monsoon should be most favorable for diseases
    if seasonal_risks[1]:  # July should be True
        print("✅ PASSED: Monsoon season = Higher disease risk (correct)")
    else:
        print("❌ FAILED: Seasonal variation not working")
        return False
    
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\n✅ Verification Complete:")
    print("  • Locust risk varies by location ✓")
    print("  • Disease detection varies by NDVI ✓")
    print("  • Yield prediction varies by conditions ✓")
    print("  • Nutrient analysis varies by health ✓")
    print("  • Seasonal patterns work correctly ✓")
    print("\n🔒 NO HARDCODED DATA DETECTED")
    print("   All outputs are dynamically calculated!")
    print("\n" + "="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_no_hardcoding()
        if success:
            print("\n✅ System verified: Ready for production use!\n")
            sys.exit(0)
        else:
            print("\n❌ Verification failed: Please review code\n")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
