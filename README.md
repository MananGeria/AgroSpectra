# AgroSpectra 🌾🛰️# AgroSpectra - WebGIS Agricultural Monitoring Platform



> **AI-Powered Precision Agriculture Platform**  ## Overview

> Integrating satellite imagery, real-time environmental data, and agricultural intelligence for smart farming decisions.

AgroSpectra is a comprehensive 9-tier WebGIS platform for precision agriculture, integrating satellite imagery, weather data, soil databases, and AI models to provide real-time crop health monitoring and pest risk prediction.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.15.0-FF4B4B.svg)](https://streamlit.io)**🌍 Global Coverage** with **🇮🇳 India-Enhanced Mode**: Works worldwide with Sentinel Hub + OpenWeatherMap. Automatically provides additional ICAR (Indian Council of Agricultural Research) insights when analyzing locations in India.

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## System Architecture

---

### Nine-Tier Architecture:

## 🌟 Overview

1. **User Interface Layer** - Streamlit-based dashboard with GPS integration

AgroSpectra is a comprehensive agricultural monitoring and analysis platform that combines:2. **Data Acquisition Layer** - Sentinel Hub, OpenWeatherMap, MOSDAC, **ICAR** APIs

- **🛰️ Satellite Imagery Analysis** - Sentinel-2 multispectral data processing3. **Geospatial Harmonization Layer** - CRS normalization, cloud masking, vegetation indices

- **🌡️ Real-Time Weather Monitoring** - OpenWeatherMap integration4. **Feature Engineering Layer** - Temporal interpolation, anomaly detection, feature fusion

- **💨 Air Quality Index (AQI)** - Pollution impact on crop health5. **AI Modeling Layer** - MobileNetV2 classifier + LSTM predictor + Fusion engine

- **🐛 Pest & Disease Tracking** - ICAR-based regional pest alerts6. **Data Storage Layer** - SQLite, GeoPackage, file storage

- **💰 Dynamic Market Pricing** - Location and season-aware crop valuations7. **Web GIS Dashboard** - Interactive maps (Folium), charts (Plotly)

- **📊 Predictive Analytics** - ML-based yield forecasting8. **Alerts and Reporting** - Rule-based alerts, PDF reports

- **🗺️ Interactive Mapping** - Folium-powered geospatial visualization9. **Feedback and Retraining** - Continuous learning from user feedback



---## Features



## 🚀 Key Features### Core Monitoring (Global)



### Environmental Monitoring- Real-time crop health classification (Healthy/Stressed/Diseased)

- **Satellite Data Processing**: Analyze NDVI, NDWI, and vegetation indices from Sentinel-2- Pest outbreak risk prediction (7-day forecast)

- **Weather Integration**: Real-time temperature, humidity, precipitation, and wind data- NDVI, NDWI, EVI vegetation indices

- **Air Quality Assessment**: PM2.5, PM10, O₃, NO₂, SO₂, CO pollutant tracking- Interactive web maps with time series analysis

- **AQI Impact Analysis**: Quantifies air pollution effects on crop health (-10% penalty) and soil quality (-15% penalty)- Automated alerts and PDF reports

- Model retraining from user feedback

### Agricultural Intelligence- **🔍 Location Search by Name** (NEW!): Search any place worldwide - no coordinates needed!

- **Regional Pest Alerts**: 100+ pest patterns across 17 Indian states with seasonal awareness- **📍 3 AOI Input Methods**: Location search, manual coordinates, or draw on map

- **Crop Recommendations**: State-specific varieties (SAU-approved), irrigation schedules, and NPK ratios

- **Soil Health Analysis**: pH, organic carbon, texture analysis with 17-state database### India-Enhanced Mode (Automatic) 🇮🇳

- **Dynamic Pricing**: Regional multipliers and seasonal variations for 12+ crops

- **Regional Pest Alerts**: ICAR-validated pest warnings by state/district

### Analytics & Insights- **Crop Recommendations**: State-specific varieties and practices

- **Yield Prediction**: ML-based forecasting with ICAR benchmark comparisons- **Soil Health Data**: District-level soil parameters from Soil Health Cards

- **Risk Assessment**: Pest risk scoring with confidence intervals- **Weather Advisories**: IMD-ICAR agromet advisories

- **Economic Valuation**: Market-aware gross value estimation- **Yield Benchmarks**: Compare your yield with state averages

- **Historical Trends**: Multi-temporal analysis capabilities

See [ICAR Setup Guide](docs/ICAR_SETUP_GUIDE.md) and [Location Search Guide](docs/LOCATION_SEARCH_GUIDE.md) for details.

---

## Technology Stack

## 📁 Project Structure

### Backend

```

AgroSpectra/- Python 3.9+

├── src/- FastAPI (REST API)

│   ├── dashboard/- GDAL 3.4+ (geospatial operations)

│   │   └── app.py                    # Main Streamlit application- Rasterio 1.2+ (raster processing)

│   ├── tier2_acquisition/            # Data Acquisition Layer- GeoPandas 0.10+ (vector data)

│   │   ├── fetch_controller.py       # Satellite & weather data fetching- TensorFlow 2.8+ / Keras (deep learning)

│   │   ├── aqi_fetcher.py           # Air quality monitoring- Celery 5.2+ (async tasks)

│   │   ├── icar_controller.py       # ICAR agricultural data- **Geopy 2.4+** (reverse geocoding for country detection)

│   │   └── market_price_fetcher.py  # Dynamic pricing system

│   ├── tier3_harmonization/          # Data Processing Layer### Database

│   ├── tier5_models/                 # ML Models Layer

│   └── tier6_storage/- SQLite 3.36+ (tabular data)

│       └── database.py              # SQLite database management- GeoPackage (spatial data)

├── data/- Redis 6.2+ (caching)

│   ├── cache/                       # API response caching

│   ├── satellite/                   # Satellite imagery storage### Frontend

│   └── storage/                     # SQLite database

├── config/                          # Configuration files- Streamlit 1.15+

├── scripts/- Folium 0.12+ (maps)

│   └── test_icar_api.py            # ICAR API testing suite- Plotly 5.11+ (charts)

├── .streamlit/- Bootstrap CSS

│   └── config.toml                 # Streamlit theme configuration

├── requirements.txt                # Python dependencies### Data Sources

├── Dockerfile                      # Docker containerization

└── docker-compose.yml              # Multi-container orchestration- Sentinel-2 imagery (via Sentinel Hub API)

```- OpenWeatherMap API

- MOSDAC (ISRO)

---- ICAR, NBSS & LUP, Bhuvan soil databases



## 🛠️ Installation## Installation



### Prerequisites### Prerequisites

- **Python**: 3.9 or higher

- **Anaconda/Miniconda**: Recommended for environment management```powershell

- **GDAL**: Required for geospatial operations# Python 3.9 or higher

python --version

### Step 1: Clone Repository

```bash# pip package manager

git clone https://github.com/MananGeria/AgroSpectra.gitpip --version

cd AgroSpectra```

```

### Setup Steps

### Step 2: Create Environment

```bash1. **Clone or navigate to project directory**

# Using Conda (Recommended)

conda create -n agrospectra python=3.9```powershell

conda activate agrospectracd "c:\Users\Manan Geria\Desktop\MyStuff\AgroSpectra"

```

# Or using venv

python -m venv agrospectra2. **Create virtual environment**

# Windows

agrospectra\Scripts\activate```powershell

# Linux/Macpython -m venv venv

source agrospectra/bin/activate.\venv\Scripts\Activate.ps1

``````



### Step 3: Install Dependencies3. **Install dependencies**

```bash

# Install geospatial libraries first (if using conda)```powershell

conda install -c conda-forge gdal rasterio geopandaspip install --upgrade pip

pip install -r requirements.txt

# Install all requirements```

pip install -r requirements.txt

```4. **Install GDAL (if not included)**



### Step 4: Configure API Keys```powershell

Create a `.env` file in the root directory:# Option 1: Using OSGeo4W installer (recommended for Windows)

# Download from: https://trac.osgeo.org/osgeo4w/

```env

# Sentinel Hub (Satellite Imagery)# Option 2: Using conda

SENTINEL_CLIENT_ID=your_client_id_here# conda install -c conda-forge gdal

SENTINEL_CLIENT_SECRET=your_client_secret_here

# Option 3: Using pip with wheel

# OpenWeatherMap (Weather & AQI)# pip install GDAL-3.4.3-cp39-cp39-win_amd64.whl

OPENWEATHER_API_KEY=your_openweather_api_key```



# ICAR Data Portal (Optional - for real-time ICAR data)5. **Configure environment variables**

USE_ICAR_REAL_API=False

ICAR_API_KEY=your_icar_api_key```powershell

AGRIMET_API_KEY=your_agrimet_api_key# Copy example config

```cp config\config.example.yaml config\config.yaml



**Getting API Keys:**# Edit config.yaml with your API credentials:

- **Sentinel Hub**: Register at [Sentinel Hub](https://www.sentinel-hub.com/)# - Sentinel Hub (client_id, client_secret, instance_id)

- **OpenWeatherMap**: Get free API key at [OpenWeatherMap](https://openweathermap.org/api)# - OpenWeatherMap API key

- **ICAR Data Portal**: Apply at [ICAR Data Portal](https://data.icar.gov.in/) (7-15 days approval)# - MOSDAC credentials (if applicable)

```

---

6. **Initialize database**

## 🎯 Usage

```powershell

### Running the Applicationpython scripts/initialize_database.py

```

```bash

# Navigate to project directory7. **Download sample datasets (optional)**

cd AgroSpectra

```powershell

# Activate environmentpython scripts/download_sample_data.py

conda activate agrospectra```



# Launch Streamlit app## Configuration

streamlit run src/dashboard/app.py

```### API Credentials Setup



The application will open in your browser at `http://localhost:8501`1. **Sentinel Hub** (Required)



### Using the Dashboard   - Register at: https://www.sentinel-hub.com/

   - Create OAuth client

1. **Enter Location**: Type city/district name or coordinates   - Add credentials to `config/config.yaml`

2. **Select Crop Type**: Choose from 12+ supported crops (Wheat, Rice, Cotton, etc.)

3. **Set Analysis Period**: Select date range for satellite data2. **OpenWeatherMap** (Required)

4. **Run Analysis**: Click "Run Analysis" to generate comprehensive report

   - Register at: https://openweathermap.org/api

### Analysis Output   - Get free API key

   - Add to `config/config.yaml`

The dashboard provides:

- **Crop Health Score**: 0-100% with AQI penalties applied3. **MOSDAC** (Optional - for Indian subcontinent)

- **Soil Quality Index**: Regional soil profiles with air quality impact   - Register at: https://www.mosdac.gov.in/

- **Pest Risk Assessment**: Real-time alerts with severity levels   - Request data access

- **Yield Prediction**: Estimated production in tons/hectare   - Add credentials to `config/config.yaml`

- **Economic Valuation**: Market-aware gross value estimation

- **Environmental Metrics**: Temperature, humidity, AQI, pollutants### Configuration File Structure

- **Interactive Maps**: Satellite imagery visualization

```yaml

---# config/config.yaml

sentinel_hub:

## 📊 Data Sources  client_id: "YOUR_CLIENT_ID"

  client_secret: "YOUR_CLIENT_SECRET"

### Real-Time APIs (Active)  instance_id: "YOUR_INSTANCE_ID"

| Source | Purpose | Update Frequency |

|--------|---------|------------------|openweathermap:

| **Sentinel Hub** | Satellite imagery (NDVI, NDWI) | 5 days |  api_key: "YOUR_API_KEY"

| **OpenWeatherMap** | Weather data | Hourly |

| **OpenWeatherMap Air Pollution** | AQI & pollutants | Hourly |mosdac:

| **Nominatim (OpenStreetMap)** | Geocoding & location | On-demand |  username: "YOUR_USERNAME"

  password: "YOUR_PASSWORD"

### Regional Databases (Enhanced)

| Database | Coverage | Details |database:

|----------|----------|---------|  sqlite_path: "data/storage/agrospectra.db"

| **ICAR Pest Patterns** | 17 states | 100+ pest types, seasonal awareness |  geopackage_path: "data/storage/agrospectra.gpkg"

| **Soil Profiles** | Pan-India | pH, OC, texture by region |

| **Crop Varieties** | State-specific | SAU-approved varieties |cache:

| **Market Prices** | 12 crops | Regional multipliers, seasonal factors |  raw_data_path: "data/cache/raw"

  processed_data_path: "data/cache/processed"

### Optional Real-Time APIs (Framework Ready)

- **ICAR Data Portal**: Pest alerts, soil health, crop advisoriesmodels:

- **IMD Agromet**: Agricultural meteorology  cnn_model_path: "models/trained/mobilenetv2_crop_health.h5"

- **Agmarknet**: Daily market prices  lstm_model_path: "models/trained/lstm_pest_risk.h5"



---alerts:

  pest_risk_threshold: 0.75

## 🧪 Testing  ndvi_anomaly_threshold: 0.2

  enable_email: false

### Test AQI Integration  enable_sms: false

```bash```

python test_aqi.py

```## Running the Application



### Test ICAR System### Start Web Dashboard

```bash

python scripts/test_icar_api.py```powershell

```streamlit run src/dashboard/app.py

```

**Expected Output:**

```The dashboard will open in your browser at `http://localhost:8501`

✅ Pest Alerts: 3 alerts retrieved

✅ Soil Health: pH 6.9, Clay Loam, 0.47% OC### Start Background Workers (Optional - for async processing)

✅ Crop Recommendations: Indrayani, Phule Radha, Sahyadri-3

✅ Location Detection: Pune, Maharashtra```powershell

```# Terminal 1: Start Redis (if using caching)

redis-server

---

# Terminal 2: Start Celery worker

## 🔧 Configurationcelery -A src.workers.celery_app worker --loglevel=info

```

### Streamlit Theme

Customize in `.streamlit/config.toml`:### Run API Server (Optional - for programmatic access)

```toml

[theme]```powershell

base = "light"uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

primaryColor = "#1f77b4"```

backgroundColor = "#ffffff"

secondaryBackgroundColor = "#f0f2f6"## Usage

textColor = "#262730"

font = "sans serif"### Basic Workflow

```

1. **Access Dashboard**: Open browser to `http://localhost:8501`

### Database Schema

AgroSpectra uses SQLite for data persistence:2. **Select Area of Interest**:

- **analysis_results**: Historical analysis records

- **user_preferences**: User settings and favorites   - Click "Use Current Location" (GPS)

- **cache_metadata**: API response caching   - OR enter coordinates manually

   - OR upload shapefile

Initialize database:

```python3. **Set Parameters**:

from src.tier6_storage.database import DatabaseManager

   - Select crop type

db = DatabaseManager()   - Choose date range

db.initialize_schema()   - Select analysis mode (real-time/historical)

```

4. **Run Analysis**: Click "Analyze Crop Health"

---

5. **View Results**:

## 🌍 Supported Regions

   - Interactive map with NDVI overlays

### Indian States (17)   - Crop health classification

- **Northern**: Punjab, Haryana, Uttar Pradesh, Himachal Pradesh   - Pest risk predictions

- **Western**: Maharashtra, Gujarat, Rajasthan   - Time series charts

- **Southern**: Karnataka, Tamil Nadu, Andhra Pradesh, Telangana, Kerala   - Alerts and recommendations

- **Eastern**: West Bengal, Bihar, Odisha

- **Central**: Madhya Pradesh, Chhattisgarh6. **Generate Report**: Download PDF report



### Crops (12+)7. **Provide Feedback**: Mark predictions as accurate/inaccurate

Wheat, Rice, Cotton, Sugarcane, Maize, Soybean, Potato, Onion, Tomato, Groundnut, Pulses, Millets

### Advanced Features

---

#### Custom Polygon Upload

## 🤝 Contributing

```python

Contributions are welcome! Please follow these steps:# Upload shapefile with field boundaries

# Supported formats: .shp, .geojson, .gpkg

1. **Fork the repository**```

2. **Create feature branch**: `git checkout -b feature/AmazingFeature`

3. **Commit changes**: `git commit -m 'Add AmazingFeature'`#### Batch Processing

4. **Push to branch**: `git push origin feature/AmazingFeature`

5. **Open Pull Request**```powershell

python scripts/batch_process.py --config batch_config.json

### Development Guidelines```

- Follow PEP 8 style guide

- Add docstrings to functions and classes#### Model Training

- Update tests for new features

- Update README if adding new functionality```powershell

# Train crop health classifier

---python src/models/train_cnn.py --data data/training/crop_health --epochs 50



## 📈 Performance# Train pest risk predictor

python src/models/train_lstm.py --data data/training/pest_sequences --epochs 100

### Caching System```

- **AQI Data**: 1-hour cache

- **Weather Data**: 30-minute cache## Project Structure

- **ICAR Data**: 7-day cache

- **Market Prices**: 1-day cache```

- **Satellite Imagery**: 30-day cacheAgroSpectra/

├── config/                      # Configuration files

### Response Times│   ├── config.yaml             # Main configuration

- **Location Search**: <1 second│   └── config.example.yaml     # Example configuration

- **AQI Fetch**: 1-2 seconds├── data/                       # Data directory

- **Weather Data**: 1-2 seconds│   ├── cache/                  # Cached data

- **Satellite Processing**: 5-10 seconds│   │   ├── raw/               # Raw downloads

- **Full Analysis**: 10-15 seconds│   │   └── processed/         # Processed data

│   ├── storage/               # Persistent storage

---│   │   ├── agrospectra.db    # SQLite database

│   │   └── agrospectra.gpkg  # GeoPackage

## 🐛 Troubleshooting│   ├── training/              # Training datasets

│   │   ├── crop_health/      # CNN training data

### Common Issues│   │   └── pest_sequences/   # LSTM training data

│   └── reports/               # Generated reports

**Issue: ModuleNotFoundError: No module named 'rasterio'**├── models/                     # AI models

```bash│   ├── trained/               # Trained model weights

# Solution: Install rasterio│   ├── train_cnn.py          # CNN training script

conda install -c conda-forge rasterio│   ├── train_lstm.py         # LSTM training script

# Or│   └── fusion.py             # Fusion engine

pip install rasterio├── notebooks/                  # Jupyter notebooks

```│   ├── data_exploration.ipynb

│   └── model_evaluation.ipynb

**Issue: GDAL import error**├── scripts/                    # Utility scripts

```bash│   ├── initialize_database.py

# Solution: Install GDAL via conda│   ├── download_sample_data.py

conda install -c conda-forge gdal│   └── batch_process.py

```├── src/                        # Source code

│   ├── tier1_ui/              # User Interface Layer

**Issue: Sentinel Hub 400 Error**│   ├── tier2_acquisition/     # Data Acquisition Layer

```│   ├── tier3_harmonization/   # Geospatial Harmonization

Check API keys in .env file│   ├── tier4_features/        # Feature Engineering

Verify Sentinel Hub subscription is active│   ├── tier5_models/          # AI Modeling

Ensure date range is within data availability│   ├── tier6_storage/         # Data Storage

```│   ├── tier7_dashboard/       # Web GIS Dashboard

│   ├── tier8_alerts/          # Alerts and Reporting

**Issue: Streamlit won't start**│   ├── tier9_feedback/        # Feedback and Retraining

```bash│   ├── api/                   # FastAPI endpoints

# Kill existing processes│   ├── workers/               # Celery workers

# Windows PowerShell:│   └── utils/                 # Utilities

Get-Process -Name "streamlit" | Stop-Process -Force├── tests/                      # Test suite

├── docker/                     # Docker files

# Linux/Mac:├── docs/                       # Documentation

pkill -f streamlit├── .env.example               # Environment variables example

```├── .gitignore                 # Git ignore file

├── docker-compose.yml         # Docker Compose configuration

---├── Dockerfile                 # Docker image definition

├── requirements.txt           # Python dependencies

## 📝 API Rate Limits└── README.md                  # This file

```

| Service | Free Tier Limit | Recommendation |

|---------|-----------------|----------------|## Docker Deployment

| **OpenWeatherMap** | 1,000 calls/day | Use caching |

| **Sentinel Hub** | 30,000 PU/month | Optimize bbox size |### Build and Run

| **Nominatim** | 1 req/second | Batch queries |

```powershell

---# Build image

docker-compose build

## 🔮 Roadmap

# Start services

### Version 2.0 (Planned)docker-compose up -d

- [ ] Multi-field comparison dashboard

- [ ] Historical trend analysis (5-year)# View logs

- [ ] Mobile app (React Native)docker-compose logs -f web

- [ ] PDF report generation

- [ ] Email notifications for pest alerts# Stop services

- [ ] Integration with IoT soil sensorsdocker-compose down

- [ ] Drone imagery processing```

- [ ] Crop insurance recommendations

### Access Services

### Version 2.5 (Future)

- [ ] Blockchain-based supply chain tracking- Web Dashboard: http://localhost:8501

- [ ] AI chatbot for farmer queries- API Documentation: http://localhost:8000/docs

- [ ] Augmented reality field scanner- Redis: localhost:6379

- [ ] Community forum for farmers

## Data Sources Setup

---

### Sentinel Hub

## 📄 License

1. Free tier: 30,000 processing units/month

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.2. Sufficient for ~100 hectares daily monitoring

3. Upgrade plans available for larger deployments

---

### OpenWeatherMap

## 👥 Authors

1. Free tier: 1,000 calls/day

- **Manan Geria** - *Initial work* - [MananGeria](https://github.com/MananGeria)2. Current weather + 5-day forecast

3. Historical data available in paid plans

---

### MOSDAC (Indian Users)

## 🙏 Acknowledgments

1. Free registration required

- **Sentinel Hub** - Satellite imagery provider2. Manual download initially (automation requires approval)

- **OpenWeatherMap** - Weather and AQI data3. Regional products optimized for India

- **ICAR** - Agricultural research databases

- **OpenStreetMap** - Geocoding services### Soil Databases

- **Streamlit** - Web framework

- **Indian Council of Agricultural Research** - Pest patterns and crop varieties1. ICAR: Download district-level data from https://krishi.icar.gov.in/

2. NBSS & LUP: Request data from https://www.nbsslup.in/

---3. Bhuvan: Access via https://bhuvan.nrsc.gov.in/



## 📧 Contact## Model Training



For questions, suggestions, or support:### Preparing Training Data

- **GitHub Issues**: [Create an issue](https://github.com/MananGeria/AgroSpectra/issues)

- **Email**: Contact through GitHub profile#### Crop Health Dataset



---```

data/training/crop_health/

## 📊 Project Statistics├── healthy/

├── stressed/

- **Total Lines of Code**: 4,000+└── diseased/

- **Python Modules**: 15+```

- **API Integrations**: 7

- **Regional Databases**: 4Each folder contains 128x128 pixel NDVI/NDWI tiles.

- **Pest Patterns**: 100+

- **Supported Crops**: 12+#### Pest Occurrence Dataset

- **Covered States**: 17

```

---data/training/pest_sequences/

├── sequences.csv           # Time series features

## ⭐ Star History└── labels.csv             # Outbreak labels

```

If you find AgroSpectra useful, please consider giving it a star on GitHub!

Format:

[![Star History Chart](https://api.star-history.com/svg?repos=MananGeria/AgroSpectra&type=Date)](https://star-history.com/#MananGeria/AgroSpectra&Date)

```csv

---date,ndvi,temperature,humidity,rainfall,lst,pest_outbreak

2024-06-01,0.65,28.5,75.2,0.0,32.1,0

**Built with ❤️ for farmers and agricultural professionals worldwide**2024-06-02,0.66,29.1,73.8,2.5,33.4,0

...

*Last Updated: November 2025*```


### Training Commands

```powershell
# Train crop health classifier (MobileNetV2)
python src/models/train_cnn.py `
  --data_dir data/training/crop_health `
  --epochs 50 `
  --batch_size 32 `
  --learning_rate 0.001 `
  --output models/trained/mobilenetv2_crop_health.h5

# Train pest risk predictor (LSTM)
python src/models/train_lstm.py `
  --data_file data/training/pest_sequences/sequences.csv `
  --labels_file data/training/pest_sequences/labels.csv `
  --sequence_length 14 `
  --epochs 100 `
  --batch_size 64 `
  --output models/trained/lstm_pest_risk.h5
```

## Troubleshooting

### Common Issues

1. **GDAL Import Error**

   ```powershell
   # Install OSGeo4W or use conda
   conda install -c conda-forge gdal rasterio geopandas
   ```

2. **Sentinel Hub Authentication Failed**

   - Verify credentials in config.yaml
   - Check instance ID is correct
   - Ensure account has processing units available

3. **Out of Memory**

   - Reduce batch size in training scripts
   - Process smaller AOI regions
   - Enable caching to reduce redundant operations

4. **Slow Performance**

   - Enable Redis caching
   - Use smaller date ranges
   - Reduce spatial resolution in config

5. **API Rate Limits**
   - OpenWeatherMap free tier: 60 calls/minute
   - Implement exponential backoff (already included)
   - Consider upgrading to paid tier

## Performance Optimization

### Caching Strategy

- Raw satellite imagery cached for 30 days
- Processed features cached for 7 days
- Weather data cached for 24 hours
- Redis for frequent database queries

### Parallel Processing

- Multiple AOIs processed concurrently
- Raster operations use GDAL's multithreading
- Batch inference for AI models

## Testing

```powershell
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_acquisition.py

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citations

### Satellite Data

- ESA Copernicus Sentinel-2 Mission
- ISRO MOSDAC Portal

### Algorithms

- MobileNetV2: Sandler et al. (2018)
- LSTM: Hochreiter & Schmidhuber (1997)
- NDVI: Tucker (1979)

## Support

- Documentation: `docs/`
- Issues: GitHub Issues
- Email: support@agrospectra.example.com

## Roadmap

### Version 1.0 (Current)

- [x] Nine-tier architecture
- [x] Basic crop health classification
- [x] Pest risk prediction
- [x] Web dashboard

### Version 1.1 (Planned)

- [ ] Mobile app (React Native)
- [ ] Multi-language support
- [ ] Irrigation scheduling
- [ ] Yield prediction

### Version 2.0 (Future)

- [ ] Drone imagery integration
- [ ] Blockchain traceability
- [ ] Market price integration
- [ ] Supply chain management

## Acknowledgments

- European Space Agency (ESA) for Sentinel-2 data
- Indian Space Research Organisation (ISRO) for MOSDAC data
- ICAR, NBSS & LUP for soil databases
- OpenWeatherMap for weather data
- Open source community for excellent libraries

---

**Built with ❤️ for sustainable agriculture**
