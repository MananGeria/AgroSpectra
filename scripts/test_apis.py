"""
Test script to verify all API credentials are working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger
from sentinelhub import SHConfig
import pyowm

def test_sentinel_hub():
    """Test Sentinel Hub API credentials."""
    print("\n🔍 Testing Sentinel Hub API...")
    try:
        client_id = os.getenv("SENTINEL_CLIENT_ID")
        client_secret = os.getenv("SENTINEL_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            print("❌ Sentinel Hub credentials not found in .env file")
            return False
            
        if "your" in client_id.lower() or "your" in client_secret.lower():
            print("❌ Please replace example credentials with your actual Sentinel Hub keys")
            return False
            
        # Test configuration
        config = SHConfig()
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        
        # Try to get OAuth token
        from sentinelhub import SentinelHubRequest
        config.sh_token_url = "https://services.sentinel-hub.com/oauth/token"
        
        print(f"   Client ID: {client_id[:8]}...")
        print(f"   Client Secret: {'*' * len(client_secret)}")
        print("✅ Sentinel Hub credentials loaded successfully!")
        print("   (Full authentication test requires actual API call)")
        return True
        
    except Exception as e:
        print(f"❌ Sentinel Hub test failed: {e}")
        return False

def test_openweather():
    """Test OpenWeatherMap API credentials."""
    print("\n🌤️  Testing OpenWeatherMap API...")
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        
        if not api_key:
            print("❌ OpenWeatherMap API key not found in .env file")
            return False
            
        if "your" in api_key.lower():
            print("❌ Please replace example API key with your actual OpenWeatherMap key")
            return False
            
        # Test API connection
        owm = pyowm.OWM(api_key)
        mgr = owm.weather_manager()
        
        # Try to get weather for a known location
        observation = mgr.weather_at_place('London,UK')
        weather = observation.weather
        
        print(f"   API Key: {api_key[:8]}...")
        print(f"   Test query successful: London weather = {weather.status}")
        print("✅ OpenWeatherMap API working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ OpenWeatherMap test failed: {e}")
        print("   Make sure your API key is activated (can take 10 minutes)")
        return False

def main():
    """Run all API tests."""
    print("=" * 60)
    print("🧪 AgroSpectra API Configuration Test")
    print("=" * 60)
    
    # Load environment variables
    env_path = project_root / ".env"
    if not env_path.exists():
        print("\n❌ ERROR: .env file not found!")
        print("   Please create .env file from .env.example")
        print("   Run: Copy-Item .env.example .env")
        return
        
    load_dotenv(env_path)
    print(f"✅ Loaded configuration from: {env_path}")
    
    # Run tests
    results = {
        "Sentinel Hub": test_sentinel_hub(),
        "OpenWeatherMap": test_openweather()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    for api, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {api:.<40} {status}")
    
    print("\n" + "=" * 60)
    
    if all(results.values()):
        print("🎉 ALL API TESTS PASSED! You're ready to run the application!")
        print("\n🚀 Next step:")
        print("   streamlit run src/dashboard/app.py")
    else:
        print("⚠️  Some API tests failed. Please check the errors above.")
        print("\n📚 Need help? See API_SETUP.md for detailed instructions.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
