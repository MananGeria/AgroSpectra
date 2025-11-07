"""
Test script for ICAR API integration
Run this to verify your API keys and endpoints are working correctly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tier2_acquisition.icar_controller import ICARController
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_icar_integration():
    """Test ICAR API integration"""
    
    print("\n" + "="*60)
    print("ICAR API Integration Test")
    print("="*60 + "\n")
    
    # Initialize controller
    controller = ICARController()
    
    # Check configuration
    print("📋 Configuration Check:")
    print(f"  Real API Enabled: {controller.use_real_api}")
    print(f"  API Key Present: {'Yes' if controller.icar_api_key else 'No'}")
    print(f"  Base URL: {controller.icar_base_url}")
    print(f"  Kisan API URL: {controller.kisan_api_url}")
    print(f"  Agrimet API URL: {controller.agrimet_api_url}")
    print()
    
    # Test parameters
    test_state = "Maharashtra"
    test_district = "Pune"
    test_crop = "rice"
    test_lat = 18.5204
    test_lon = 73.8567
    
    # Test 1: Pest Alerts
    print("🐛 Test 1: Fetching Pest Alerts...")
    print(f"  Location: {test_district}, {test_state}")
    print(f"  Crop: {test_crop}")
    
    try:
        pest_alerts = controller.fetch_pest_alerts(test_state, test_district, test_crop)
        
        if pest_alerts:
            print(f"  ✅ Success! Retrieved {len(pest_alerts)} alerts")
            print(f"  Data Source: {pest_alerts[0].get('source', 'Unknown')}")
            
            # Show first alert
            if len(pest_alerts) > 0:
                alert = pest_alerts[0]
                print(f"\n  Sample Alert:")
                print(f"    Pest: {alert.get('pest_name')}")
                print(f"    Severity: {alert.get('severity')}")
                print(f"    Confidence: {alert.get('confidence', 0)*100:.0f}%")
        else:
            print("  ⚠️ No alerts retrieved")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print()
    
    # Test 2: Soil Health
    print("🌱 Test 2: Fetching Soil Health Data...")
    print(f"  Location: {test_district}, {test_state}")
    
    try:
        soil_data = controller.fetch_soil_health_data(test_state, test_district)
        
        if soil_data:
            print(f"  ✅ Success! Retrieved soil data")
            print(f"  Data Source: {soil_data.get('source', 'Unknown')}")
            print(f"\n  Soil Parameters:")
            print(f"    pH: {soil_data.get('ph', 'N/A')}")
            print(f"    Organic Carbon: {soil_data.get('organic_carbon', 'N/A')}")
            print(f"    Texture: {soil_data.get('texture', 'N/A')}")
            print(f"    Nitrogen: {soil_data.get('nitrogen', 'N/A')}")
        else:
            print("  ⚠️ No soil data retrieved")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print()
    
    # Test 3: Crop Recommendations
    print("🌾 Test 3: Fetching Crop Recommendations...")
    print(f"  Location: {test_state}")
    print(f"  Crop: {test_crop}")
    print(f"  Season: Kharif")
    
    try:
        recommendations = controller.fetch_crop_recommendations(test_state, test_crop, "Kharif")
        
        if recommendations:
            print(f"  ✅ Success! Retrieved recommendations")
            print(f"  Data Source: {recommendations.get('source', 'Unknown')}")
            print(f"\n  Recommendations:")
            varieties = recommendations.get('recommended_varieties', [])
            if varieties:
                print(f"    Varieties: {', '.join(varieties[:3])}")
            print(f"    NPK Ratio: {recommendations.get('npk_ratio', 'N/A')}")
            print(f"    Expected Yield: {recommendations.get('estimated_yield', 'N/A')}")
        else:
            print("  ⚠️ No recommendations retrieved")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print()
    
    # Test 4: Location Details
    print("📍 Test 4: Reverse Geocoding...")
    print(f"  Coordinates: {test_lat}, {test_lon}")
    
    try:
        location = controller.get_location_details(test_lat, test_lon)
        
        if location:
            print(f"  ✅ Success!")
            print(f"    State: {location.get('state')}")
            print(f"    District: {location.get('district')}")
            print(f"    Country: {location.get('country')}")
            print(f"    Is India: {location.get('is_india')}")
        else:
            print("  ⚠️ No location data retrieved")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
    
    print()
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    if controller.use_real_api and controller.icar_api_key:
        print("✅ Real API mode enabled")
        print("📡 System will try to fetch from ICAR APIs")
        print("🔄 Falls back to regional database if API fails")
    else:
        print("📊 Using regional database (Real API disabled)")
        print("\nTo enable real API:")
        print("1. Set environment variable: USE_ICAR_REAL_API=true")
        print("2. Add your API key: ICAR_API_KEY=your_key")
        print("3. See docs/ICAR_API_INTEGRATION.md for details")
    
    print("\n" + "="*60 + "\n")


def check_environment():
    """Check if environment variables are set"""
    
    print("\n" + "="*60)
    print("Environment Variables Check")
    print("="*60 + "\n")
    
    env_vars = {
        'USE_ICAR_REAL_API': 'Enable/disable real API',
        'ICAR_API_KEY': 'ICAR Data Portal key',
        'ICAR_API_BASE_URL': 'ICAR API endpoint',
        'KISAN_API_URL': 'Kisan Portal endpoint',
        'AGRIMET_API_URL': 'IMD Agromet endpoint',
        'OPENWEATHER_API_KEY': 'OpenWeatherMap key (in use)'
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            # Mask API keys
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = value[:8] + '...' if len(value) > 8 else '***'
            else:
                display_value = value
            print(f"✅ {var:25} = {display_value}")
        else:
            print(f"❌ {var:25} = Not Set")
        print(f"   Description: {description}")
        print()
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Check environment first
    check_environment()
    
    # Run tests
    test_icar_integration()
    
    print("\n💡 Next Steps:")
    print("1. If using real APIs: Ensure your API keys are valid")
    print("2. Check docs/ICAR_API_INTEGRATION.md for registration process")
    print("3. Monitor logs for API call details")
    print("4. Regional database works as fallback automatically\n")
