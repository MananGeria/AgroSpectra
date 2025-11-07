# AgroSpectra - WebGIS Agricultural Monitoring Platform

## Overview

AgroSpectra is a comprehensive 9-tier WebGIS platform for precision agriculture, integrating satellite imagery, weather data, soil databases, and AI models to provide real-time crop health monitoring and pest risk prediction.

## System Architecture

### Nine-Tier Architecture:

1. **User Interface Layer** - Streamlit-based dashboard with GPS integration
2. **Data Acquisition Layer** - Sentinel Hub, OpenWeatherMap, MOSDAC APIs
3. **Geospatial Harmonization Layer** - CRS normalization, cloud masking, vegetation indices
4. **Feature Engineering Layer** - Temporal interpolation, anomaly detection, feature fusion
5. **AI Modeling Layer** - MobileNetV2 classifier + LSTM predictor + Fusion engine
6. **Data Storage Layer** - SQLite, GeoPackage, file storage
7. **Web GIS Dashboard** - Interactive maps (Folium), charts (Plotly)
8. **Alerts and Reporting** - Rule-based alerts, PDF reports
9. **Feedback and Retraining** - Continuous learning from user feedback

## Features

- Real-time crop health classification (Healthy/Stressed/Diseased)
- Pest outbreak risk prediction (7-day forecast)
- NDVI, NDWI, EVI vegetation indices
- Interactive web maps with time series analysis
- Automated alerts and PDF reports
- Model retraining from user feedback

## Technology Stack

### Backend

- Python 3.9+
- FastAPI (REST API)
- GDAL 3.4+ (geospatial operations)
- Rasterio 1.2+ (raster processing)
- GeoPandas 0.10+ (vector data)
- TensorFlow 2.8+ / Keras (deep learning)
- Celery 5.2+ (async tasks)

### Database

- SQLite 3.36+ (tabular data)
- GeoPackage (spatial data)
- Redis 6.2+ (caching)

### Frontend

- Streamlit 1.15+
- Folium 0.12+ (maps)
- Plotly 5.11+ (charts)
- Bootstrap CSS

### Data Sources

- Sentinel-2 imagery (via Sentinel Hub API)
- OpenWeatherMap API
- MOSDAC (ISRO)
- ICAR, NBSS & LUP, Bhuvan soil databases

## Installation

### Prerequisites

```powershell
# Python 3.9 or higher
python --version

# pip package manager
pip --version
```

### Setup Steps

1. **Clone or navigate to project directory**

```powershell
cd "c:\Users\Manan Geria\Desktop\MyStuff\AgroSpectra"
```

2. **Create virtual environment**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install GDAL (if not included)**

```powershell
# Option 1: Using OSGeo4W installer (recommended for Windows)
# Download from: https://trac.osgeo.org/osgeo4w/

# Option 2: Using conda
# conda install -c conda-forge gdal

# Option 3: Using pip with wheel
# pip install GDAL-3.4.3-cp39-cp39-win_amd64.whl
```

5. **Configure environment variables**

```powershell
# Copy example config
cp config\config.example.yaml config\config.yaml

# Edit config.yaml with your API credentials:
# - Sentinel Hub (client_id, client_secret, instance_id)
# - OpenWeatherMap API key
# - MOSDAC credentials (if applicable)
```

6. **Initialize database**

```powershell
python scripts/initialize_database.py
```

7. **Download sample datasets (optional)**

```powershell
python scripts/download_sample_data.py
```

## Configuration

### API Credentials Setup

1. **Sentinel Hub** (Required)

   - Register at: https://www.sentinel-hub.com/
   - Create OAuth client
   - Add credentials to `config/config.yaml`

2. **OpenWeatherMap** (Required)

   - Register at: https://openweathermap.org/api
   - Get free API key
   - Add to `config/config.yaml`

3. **MOSDAC** (Optional - for Indian subcontinent)
   - Register at: https://www.mosdac.gov.in/
   - Request data access
   - Add credentials to `config/config.yaml`

### Configuration File Structure

```yaml
# config/config.yaml
sentinel_hub:
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"
  instance_id: "YOUR_INSTANCE_ID"

openweathermap:
  api_key: "YOUR_API_KEY"

mosdac:
  username: "YOUR_USERNAME"
  password: "YOUR_PASSWORD"

database:
  sqlite_path: "data/storage/agrospectra.db"
  geopackage_path: "data/storage/agrospectra.gpkg"

cache:
  raw_data_path: "data/cache/raw"
  processed_data_path: "data/cache/processed"

models:
  cnn_model_path: "models/trained/mobilenetv2_crop_health.h5"
  lstm_model_path: "models/trained/lstm_pest_risk.h5"

alerts:
  pest_risk_threshold: 0.75
  ndvi_anomaly_threshold: 0.2
  enable_email: false
  enable_sms: false
```

## Running the Application

### Start Web Dashboard

```powershell
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Start Background Workers (Optional - for async processing)

```powershell
# Terminal 1: Start Redis (if using caching)
redis-server

# Terminal 2: Start Celery worker
celery -A src.workers.celery_app worker --loglevel=info
```

### Run API Server (Optional - for programmatic access)

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Basic Workflow

1. **Access Dashboard**: Open browser to `http://localhost:8501`

2. **Select Area of Interest**:

   - Click "Use Current Location" (GPS)
   - OR enter coordinates manually
   - OR upload shapefile

3. **Set Parameters**:

   - Select crop type
   - Choose date range
   - Select analysis mode (real-time/historical)

4. **Run Analysis**: Click "Analyze Crop Health"

5. **View Results**:

   - Interactive map with NDVI overlays
   - Crop health classification
   - Pest risk predictions
   - Time series charts
   - Alerts and recommendations

6. **Generate Report**: Download PDF report

7. **Provide Feedback**: Mark predictions as accurate/inaccurate

### Advanced Features

#### Custom Polygon Upload

```python
# Upload shapefile with field boundaries
# Supported formats: .shp, .geojson, .gpkg
```

#### Batch Processing

```powershell
python scripts/batch_process.py --config batch_config.json
```

#### Model Training

```powershell
# Train crop health classifier
python src/models/train_cnn.py --data data/training/crop_health --epochs 50

# Train pest risk predictor
python src/models/train_lstm.py --data data/training/pest_sequences --epochs 100
```

## Project Structure

```
AgroSpectra/
├── config/                      # Configuration files
│   ├── config.yaml             # Main configuration
│   └── config.example.yaml     # Example configuration
├── data/                       # Data directory
│   ├── cache/                  # Cached data
│   │   ├── raw/               # Raw downloads
│   │   └── processed/         # Processed data
│   ├── storage/               # Persistent storage
│   │   ├── agrospectra.db    # SQLite database
│   │   └── agrospectra.gpkg  # GeoPackage
│   ├── training/              # Training datasets
│   │   ├── crop_health/      # CNN training data
│   │   └── pest_sequences/   # LSTM training data
│   └── reports/               # Generated reports
├── models/                     # AI models
│   ├── trained/               # Trained model weights
│   ├── train_cnn.py          # CNN training script
│   ├── train_lstm.py         # LSTM training script
│   └── fusion.py             # Fusion engine
├── notebooks/                  # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── scripts/                    # Utility scripts
│   ├── initialize_database.py
│   ├── download_sample_data.py
│   └── batch_process.py
├── src/                        # Source code
│   ├── tier1_ui/              # User Interface Layer
│   ├── tier2_acquisition/     # Data Acquisition Layer
│   ├── tier3_harmonization/   # Geospatial Harmonization
│   ├── tier4_features/        # Feature Engineering
│   ├── tier5_models/          # AI Modeling
│   ├── tier6_storage/         # Data Storage
│   ├── tier7_dashboard/       # Web GIS Dashboard
│   ├── tier8_alerts/          # Alerts and Reporting
│   ├── tier9_feedback/        # Feedback and Retraining
│   ├── api/                   # FastAPI endpoints
│   ├── workers/               # Celery workers
│   └── utils/                 # Utilities
├── tests/                      # Test suite
├── docker/                     # Docker files
├── docs/                       # Documentation
├── .env.example               # Environment variables example
├── .gitignore                 # Git ignore file
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Docker Deployment

### Build and Run

```powershell
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

### Access Services

- Web Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- Redis: localhost:6379

## Data Sources Setup

### Sentinel Hub

1. Free tier: 30,000 processing units/month
2. Sufficient for ~100 hectares daily monitoring
3. Upgrade plans available for larger deployments

### OpenWeatherMap

1. Free tier: 1,000 calls/day
2. Current weather + 5-day forecast
3. Historical data available in paid plans

### MOSDAC (Indian Users)

1. Free registration required
2. Manual download initially (automation requires approval)
3. Regional products optimized for India

### Soil Databases

1. ICAR: Download district-level data from https://krishi.icar.gov.in/
2. NBSS & LUP: Request data from https://www.nbsslup.in/
3. Bhuvan: Access via https://bhuvan.nrsc.gov.in/

## Model Training

### Preparing Training Data

#### Crop Health Dataset

```
data/training/crop_health/
├── healthy/
├── stressed/
└── diseased/
```

Each folder contains 128x128 pixel NDVI/NDWI tiles.

#### Pest Occurrence Dataset

```
data/training/pest_sequences/
├── sequences.csv           # Time series features
└── labels.csv             # Outbreak labels
```

Format:

```csv
date,ndvi,temperature,humidity,rainfall,lst,pest_outbreak
2024-06-01,0.65,28.5,75.2,0.0,32.1,0
2024-06-02,0.66,29.1,73.8,2.5,33.4,0
...
```

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
