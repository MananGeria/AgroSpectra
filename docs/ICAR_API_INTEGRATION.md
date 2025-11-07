# ICAR Real-Time API Integration Guide

## Overview

This guide explains how to integrate AgroSpectra with real-time ICAR (Indian Council of Agricultural Research) APIs for fetching live agricultural data.

## Available ICAR Data Sources

### 1. **ICAR Data Portal**

- **URL**: https://data.icar.gov.in/
- **Data Available**: Research datasets, crop statistics, pest surveillance
- **Status**: Registration required for API access

### 2. **Soil Health Card Portal**

- **URL**: https://soilhealth.dac.gov.in/
- **Data Available**: District-wise soil test results, nutrient status
- **API**: Currently limited public API; data available via web scraping or partnership
- **Alternative**: State Agricultural Department APIs

### 3. **IMD Agromet Advisory**

- **URL**: https://www.imdagrimet.gov.in/
- **Data Available**: Weather-based agricultural advisories, district-wise forecasts
- **API**: Available through IMD registration

### 4. **Kisan Portal (farmer.gov.in)**

- **URL**: https://farmer.gov.in/
- **Data Available**: Pest alerts, market prices, crop advisories
- **API**: Contact Ministry of Agriculture for access

### 5. **M-Kisan Portal**

- **URL**: https://mkisan.gov.in/
- **Data Available**: SMS-based advisories, pest warnings
- **API**: Available for government/research institutions

### 6. **NIPHM (Pest Management)**

- **URL**: https://niphm.gov.in/
- **Data Available**: Pest surveillance, IPM recommendations
- **Contact**: National Institute of Plant Health Management

---

## How to Get API Access

### Step 1: Register with ICAR Data Portal

1. Visit: https://data.icar.gov.in/
2. Click on "Register" → Fill application form
3. Purpose: Research/Agricultural Technology Development
4. Wait for approval (typically 7-15 days)
5. Receive API key via email

### Step 2: Register with IMD Agromet

1. Visit: https://www.imdagrimet.gov.in/
2. Navigate to "Data Services" → "API Access"
3. Submit request with project details
4. Approval process: 2-4 weeks
5. Documentation provided after approval

### Step 3: Alternative - State Agricultural Department APIs

Many states have their own APIs:

- **Punjab**: https://agripb.gov.in/
- **Maharashtra**: https://krishi.maharashtra.gov.in/
- **Karnataka**: https://raitamitra.karnataka.gov.in/
- **Tamil Nadu**: https://www.tn.gov.in/scheme/data_view/48

Contact respective State Agricultural Universities for API access.

---

## Configuration Setup

### Step 1: Set Environment Variables

Create or edit `.env` file in your project root:

```bash
# Enable real-time API fetching
USE_ICAR_REAL_API=true

# ICAR Data Portal
ICAR_API_KEY=your_icar_api_key_here
ICAR_API_BASE_URL=https://data.icar.gov.in/api/v1

# Kisan Portal (if available)
KISAN_API_URL=https://api.farmer.gov.in/v1
KISAN_API_KEY=your_kisan_api_key

# IMD Agromet
AGRIMET_API_URL=https://www.imdagrimet.gov.in/api
AGRIMET_API_KEY=your_imd_api_key

# Soil Health Portal (if you have access)
SOIL_HEALTH_API_KEY=your_soil_health_key
```

### Step 2: Install Required Package (if not already installed)

```powershell
pip install python-dotenv
```

### Step 3: Load Environment Variables in Code

The `icar_controller.py` already uses `os.getenv()` to read these variables:

```python
self.icar_api_key = os.getenv('ICAR_API_KEY', '')
self.use_real_api = os.getenv('USE_ICAR_REAL_API', 'false').lower() == 'true'
```

---

## API Integration Details

### Current Implementation Status

#### ✅ **Ready for Integration**

- Pest alerts fetching
- Soil health data fetching
- Crop advisory fetching
- Automatic fallback to regional database if API fails

#### 📋 **Integration Points**

**1. Pest Alerts (`_fetch_real_pest_alerts`)**

```python
# API Endpoint Structure (adapt based on actual API docs)
GET /api/pest-surveillance/alerts
Parameters:
  - state: String (e.g., "Maharashtra")
  - district: String (e.g., "Pune")
  - crop: String (e.g., "cotton")
  - date: String (YYYY-MM-DD)

Headers:
  - Authorization: Bearer {API_KEY}
  - Content-Type: application/json
```

**2. Soil Health (`_fetch_real_soil_health`)**

```python
# API Endpoint Structure
GET https://soilhealth.dac.gov.in/api/soil-data
Parameters:
  - state: String
  - district: String
  - latitude: Float (optional)
  - longitude: Float (optional)

Response Format (expected):
{
  "soil_data": {
    "pH": 7.2,
    "organic_carbon": "0.45%",
    "nitrogen": "Medium",
    "phosphorus": "25 kg/ha",
    ...
  }
}
```

**3. Crop Advisory (`_fetch_real_crop_advisory`)**

```python
# API Endpoint Structure
GET https://www.imdagrimet.gov.in/api/advisory
Parameters:
  - state: String
  - district: String
  - crop: String
  - week: Integer (ISO week number)

Response Format:
{
  "advisory": {
    "varieties": ["PBW-725", "HD-3086"],
    "sowing_time": "November 1-15",
    "fertilizer_recommendation": "120:60:40 NPK",
    ...
  }
}
```

---

## Testing the Integration

### Method 1: Test with Environment Variables

```powershell
# Set test environment
$env:USE_ICAR_REAL_API="true"
$env:ICAR_API_KEY="test_key_12345"

# Run Streamlit
streamlit run src/dashboard/app.py
```

### Method 2: Check Logs

The code logs API calls:

```
[INFO] ICAR Controller initialized (Real API: True)
[INFO] Successfully fetched pest data from ICAR API
[WARNING] ICAR API returned status 401 (authentication failed)
```

### Method 3: Verify Data Source

In the UI, check the data source labels:

- **Real API**: "ICAR Real-time Data"
- **Fallback**: "Regional Soil Database" / "State Agricultural Department Advisory"

---

## API Response Parsing

The code includes parsing functions that adapt to different API structures:

### Flexible Parsing Example

```python
def _parse_icar_pest_response(self, data: Dict) -> List[Dict]:
    # Handles multiple possible field names
    pest_data = data.get('alerts', []) or data.get('data', [])

    for item in pest_data:
        pest_name = item.get('pest_name') or item.get('pestName')
        # Fallback to multiple field name variations
```

**You need to adapt these functions** based on actual API documentation when you receive access.

---

## Troubleshooting

### Issue 1: API Key Not Working

**Solution**:

- Verify key is active: Contact ICAR support
- Check environment variable is loaded: `print(os.getenv('ICAR_API_KEY'))`
- Ensure no extra spaces in `.env` file

### Issue 2: 401 Unauthorized

**Causes**:

- Invalid API key
- API key expired
- Wrong authorization header format

**Fix**: Check API documentation for correct header format

### Issue 3: Timeout Errors

**Solution**:

- Increase timeout: Modify `self.api_timeout = 10` to higher value
- Check network connectivity
- Verify API endpoint is accessible

### Issue 4: Data Format Mismatch

**Solution**:

- Log raw API response: `print(response.json())`
- Update parsing functions to match actual format
- Contact API provider for documentation

---

## Free/Open Alternatives (No API Key Required)

While waiting for ICAR API access, you can use:

### 1. **OpenWeatherMap Agri API** (Currently in use for weather)

- URL: https://openweathermap.org/api/agro
- Features: NDVI, EVI, soil temperature
- Limitations: No pest alerts or soil nutrients

### 2. **Copernicus Open Access Hub** (Satellite data)

- URL: https://scihub.copernicus.eu/
- Features: Sentinel satellite imagery (already integrated)
- Free for research use

### 3. **NASA POWER** (Weather for agriculture)

- URL: https://power.larc.nasa.gov/
- Features: Solar data, temperature, precipitation
- No API key required

### 4. **ISRO Bhuvan** (Indian geospatial data)

- URL: https://bhuvan.nrsc.gov.in/
- Features: Land use, crop mapping
- Registration required but free

---

## Production Deployment Checklist

Before deploying with real APIs:

- [ ] Obtain API keys from all required sources
- [ ] Test API endpoints with sample requests
- [ ] Implement rate limiting (most APIs have limits)
- [ ] Add retry logic for failed requests
- [ ] Set up monitoring for API availability
- [ ] Cache responses to reduce API calls
- [ ] Document API costs (if any)
- [ ] Add graceful degradation to fallback data
- [ ] Test error handling thoroughly
- [ ] Set up API usage alerts

---

## Current Status Summary

| Data Type         | Source              | API Available | Integration Status                |
| ----------------- | ------------------- | ------------- | --------------------------------- |
| Pest Alerts       | ICAR + State Depts  | Partial       | ✅ Code ready, needs API key      |
| Soil Health       | Soil Health Card    | Limited       | ✅ Code ready, needs partnership  |
| Crop Advisory     | IMD Agromet         | Yes           | ✅ Code ready, needs registration |
| Weather Data      | OpenWeatherMap      | Yes           | ✅ Already integrated             |
| Satellite Imagery | Copernicus/Sentinel | Yes           | ✅ Already integrated             |
| AQI Data          | OpenWeatherMap      | Yes           | ✅ Already integrated             |
| Market Prices     | Agmarknet           | Yes (web)     | ⏳ Can be added                   |

---

## Next Steps

1. **Apply for API Access**: Start registration process with ICAR and IMD
2. **Contact State Departments**: Reach out to your state's agricultural department
3. **Join Research Programs**: ICAR often provides API access to research partners
4. **Pilot Program**: Consider proposing a pilot program with Ministry of Agriculture
5. **Academic Partnership**: If affiliated with university, leverage institutional access

---

## Support & Resources

- **ICAR Support**: support@icar.gov.in
- **IMD Agromet**: agrimet@imd.gov.in
- **Technical Issues**: Check logs in `logs/` directory
- **Community**: Indian Agricultural Research Community forums

---

## Code Modification Guide

When you receive actual API documentation, you'll need to modify:

### 1. Update API Endpoints

```python
# In icar_controller.py __init__ method
self.icar_base_url = os.getenv('ICAR_API_BASE_URL', 'ACTUAL_URL_HERE')
```

### 2. Adjust Request Headers

```python
# In _fetch_real_pest_alerts or other fetch methods
headers = {
    'Authorization': 'ACTUAL_AUTH_FORMAT',  # Could be API-Key: xxx
    'X-Custom-Header': 'value'  # Add any custom headers
}
```

### 3. Update Response Parsing

```python
# In _parse_icar_pest_response
# Match the actual JSON structure from API docs
pest_name = item['actual_field_name_from_docs']
```

---

## Contact for API Access

**For quickest results, contact:**

1. ICAR Krishi Vigyan Kendras (KVK) in your region
2. State Agricultural University IT departments
3. Ministry of Agriculture - Digital Agriculture team
4. ICAR-Indian Agricultural Statistics Research Institute (IASRI)

**This integration framework is ready - you just need the API keys! 🚀**
