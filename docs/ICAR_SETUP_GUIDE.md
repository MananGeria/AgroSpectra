# ICAR Integration Setup Guide

## Overview

AgroSpectra now includes **ICAR (Indian Council of Agricultural Research)** integration to provide enhanced agricultural insights specifically for Indian locations. This feature is **completely optional** and does not affect the app's global functionality.

## 🌍 How It Works

### Global Coverage (Default)

- Works **anywhere in the world**
- Uses Sentinel Hub satellite data + OpenWeatherMap
- Provides comprehensive crop monitoring

### India-Enhanced Mode (Automatic)

- **Automatically detects** when your GPS coordinates are in India
- Fetches **additional ICAR data** from Indian agricultural databases
- Shows enhanced insights:
  - Regional pest alerts
  - State-specific crop varieties
  - ICAR crop recommendations
  - Soil health card data
  - IMD weather advisories

### Example Scenarios

| Location          | Data Sources Used                        | Features Available                  |
| ----------------- | ---------------------------------------- | ----------------------------------- |
| Iowa, USA         | Sentinel Hub + OpenWeatherMap            | Full monitoring (standard)          |
| Punjab, India     | Sentinel Hub + OpenWeatherMap + **ICAR** | Full monitoring + **India bonuses** |
| São Paulo, Brazil | Sentinel Hub + OpenWeatherMap            | Full monitoring (standard)          |

## 🚀 Quick Start (No Setup Required!)

ICAR integration is **enabled by default** and requires **zero configuration**. Just run the app:

```bash
streamlit run src/dashboard/app.py
```

When you select a location in India, you'll automatically see the 🇮🇳 badge and enhanced data sections.

## 📊 ICAR Data Sources

Since ICAR does not provide official APIs, the app uses **simulated data** based on:

1. **Regional Pest Patterns**: Common pests by state and crop type
2. **State Agricultural Universities**: Recommended varieties and practices
3. **ICAR Guidelines**: Standard fertilizer and irrigation recommendations
4. **Soil Health Averages**: District-level soil characteristics

### Future Enhancement: Real ICAR Data

To use **real ICAR data** (requires manual setup), you can:

#### Option 1: Download ICAR Datasets

1. Visit [data.icar.gov.in](https://data.icar.gov.in/)
2. Register for free account (email-based)
3. Download district-wise datasets:
   - Pest alerts (CSV)
   - Crop advisories (PDF → convert to CSV)
   - Soil health cards (Excel)
4. Place datasets in `data/icar/` folder:
   ```
   data/icar/
   ├── pest_alerts/
   │   ├── punjab_alerts.csv
   │   ├── maharashtra_alerts.csv
   │   └── ...
   ├── crop_advisories/
   │   ├── wheat_recommendations.csv
   │   └── ...
   └── soil_health/
       ├── district_soil_data.csv
       └── ...
   ```

#### Option 2: Use ICAR Web Portals (Screen Scraping)

1. **Krishi Portal**: [krishiportal.icar.gov.in](https://krishiportal.icar.gov.in/)
   - Crop advisories by state
   - Pest management guidelines
2. **NBAIR (Pest Alerts)**: [www.nbair.res.in](https://www.nbair.res.in/)

   - Real-time pest warnings
   - Biological control recommendations

3. **CRIDA (Dryland Agriculture)**: [www.crida.in](https://www.crida.in/)
   - Soil moisture data
   - Drought management

**Note**: Web scraping requires additional libraries and compliance with website terms of service.

## ⚙️ Configuration

### Environment Variables (.env)

The following settings control ICAR integration:

```properties
# ICAR Configuration
ICAR_ENABLED=true                    # Enable/disable ICAR features
ICAR_DATA_PATH=data/icar/            # Path to ICAR datasets
ICAR_CACHE_EXPIRY_DAYS=7             # Cache validity (days)
ICAR_UPDATE_FREQUENCY=weekly         # Update schedule
```

### Disable ICAR (if needed)

To completely disable ICAR features:

1. Edit `.env` file:

   ```properties
   ICAR_ENABLED=false
   ```

2. Restart the app

The app will continue to work globally with Sentinel Hub + OpenWeatherMap only.

## 🧪 Testing

### Test with Indian Location

1. Run the app
2. Select a location in India (e.g., Punjab: 30.73°N, 75.85°E)
3. Run analysis
4. Look for:
   - 🇮🇳 "ICAR Enhancement Active" badge
   - "ICAR Pest Alerts" section
   - "ICAR Crop Recommendations" section
   - "ICAR Soil Health Card Data" section
   - "IMD-ICAR Weather Advisory" section

### Test with Non-Indian Location

1. Select a location outside India (e.g., USA: 41.88°N, -93.09°W)
2. Run analysis
3. Verify:
   - No ICAR sections appear
   - App works normally with global data
   - All features functional

## 📈 ICAR Data Coverage

### Current Implementation (Simulated Data)

| Data Type        | Coverage       | Update Frequency      | Source              |
| ---------------- | -------------- | --------------------- | ------------------- |
| Pest Alerts      | 8 major states | Real-time (simulated) | Regional patterns   |
| Crop Varieties   | 5 major crops  | Static                | SAU recommendations |
| Soil Health      | All districts  | Static (simulated)    | Average values      |
| Weather Advisory | All states     | Daily (simulated)     | IMD patterns        |

### With Real ICAR Datasets (Future)

| Data Type        | Coverage       | Update Frequency | Source                    |
| ---------------- | -------------- | ---------------- | ------------------------- |
| Pest Alerts      | 28 states      | Weekly           | NBAIR, NCIPM              |
| Crop Varieties   | 100+ crops     | Seasonal         | SAUs, ICAR institutes     |
| Soil Health      | 700+ districts | Annual           | Soil Health Card database |
| Weather Advisory | All states     | Daily            | IMD-ICAR Agromet          |

## 🔐 Data Privacy & Licensing

### ICAR Data Terms

- **Free for non-commercial use**: Academic research, personal projects
- **Commercial use**: Requires permission from ICAR
  - Email: director.icar@gov.in
  - Specify use case and scale
  - May involve licensing fees

### AgroSpectra Compliance

- Current implementation uses **simulated/generalized data** (no licensing issues)
- **Real ICAR datasets**: User responsible for compliance
- App does **not store or redistribute** ICAR data
- All data is cached locally only

## 🐛 Troubleshooting

### Issue: "ICAR enhancement unavailable"

**Cause**: Error fetching location or ICAR data

**Fix**:

1. Check internet connection (reverse geocoding requires API access)
2. Verify `.env` has `ICAR_ENABLED=true`
3. Check logs: `logs/agrospectra.log`

### Issue: No ICAR sections for India location

**Cause**: Country detection failed

**Fix**:

1. Ensure coordinates are accurate
2. Try different coordinates in same state
3. Check reverse geocoding service (Nominatim) is accessible

### Issue: ICAR data seems outdated

**Cause**: Cache not expired

**Fix**:

1. Clear cache: Delete `data/cache/icar/` folder
2. Adjust `ICAR_CACHE_EXPIRY_DAYS` in `.env`
3. Restart app

## 📝 Example Usage

### Python Script (Direct API)

```python
from tier2_acquisition.icar_controller import ICARController, ICARDataEnhancer

# Initialize controller
icar = ICARController()

# Get location details
location = icar.get_location_details(lat=30.73, lon=75.85)
print(f"Location: {location['district']}, {location['state']}")
print(f"Is India: {location['is_india']}")

# Fetch pest alerts (if in India)
if location['is_india']:
    alerts = icar.fetch_pest_alerts(
        state=location['state'],
        district=location['district'],
        crop_type='wheat'
    )
    for alert in alerts:
        print(f"Pest: {alert['pest_name']} - Severity: {alert['severity']}")

# Get crop recommendations
recommendations = icar.fetch_crop_recommendations(
    state=location['state'],
    crop_type='wheat',
    season='Rabi'
)
print(f"Recommended varieties: {recommendations['recommended_varieties']}")

# Enhance base predictions
base_pest_risk = 0.6
enhanced = ICARDataEnhancer.enhance_pest_risk(base_pest_risk, alerts)
print(f"Enhanced risk: {enhanced['risk_score']} (confidence: {enhanced['confidence']})")
```

## 🎯 Roadmap

### Phase 1: Foundation (Current)

- ✅ Country detection
- ✅ Simulated ICAR data
- ✅ UI integration
- ✅ Documentation

### Phase 2: Real Data Integration (Planned)

- ⏳ ICAR dataset downloads
- ⏳ CSV/Excel parsers
- ⏳ Automatic updates
- ⏳ Data validation

### Phase 3: Advanced Features (Future)

- ⏳ Web scraping for real-time alerts
- ⏳ Multi-language support (Hindi, Telugu, Tamil)
- ⏳ State-specific crop calendars
- ⏳ MSP price tracking

## 📞 Support

### For AgroSpectra Issues

- GitHub Issues: https://github.com/MananGeria/AgroSpectra/issues
- Email: [Your contact email]

### For ICAR Data Access

- ICAR Website: https://icar.org.in/
- Data Portal: https://data.icar.gov.in/
- Helpdesk: data-helpdesk@icar.gov.in

## 📚 Additional Resources

- **ICAR Official Site**: https://icar.org.in/
- **Krishi Portal**: https://krishiportal.icar.gov.in/
- **Soil Health Card**: https://soilhealth.dac.gov.in/
- **IMD Agromet**: https://mausam.imd.gov.in/imd_latest/contents/agromet_advisory.php
- **NBAIR Pest Alerts**: https://www.nbair.res.in/
- **State Agricultural Universities**: https://icar.org.in/content/state-agricultural-universities

---

**Version**: 1.0  
**Last Updated**: 2024  
**Maintained By**: AgroSpectra Team
