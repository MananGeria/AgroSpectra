"""
Database Schema and Initialization
SQLite database for storing agricultural monitoring data
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from loguru import logger


class DatabaseManager:
    """Manage SQLite database operations"""
    
    def __init__(self, db_path: str = "data/storage/agrospectra.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def initialize_schema(self):
        """Create database tables"""
        logger.info("Initializing database schema")
        
        cursor = self.conn.cursor()
        
        # Satellite data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satellite_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bbox TEXT NOT NULL,
                ndvi_mean REAL,
                ndvi_std REAL,
                ndwi_mean REAL,
                ndwi_std REAL,
                evi_mean REAL,
                evi_std REAL,
                cloud_cover REAL,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Weather data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                rainfall_mm REAL,
                windspeed_ms REAL,
                lst_c REAL,
                cloud_cover_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Soil data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS soil_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                polygon_id TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                nitrogen_kgha REAL,
                phosphorus_kgha REAL,
                potassium_kgha REAL,
                ph REAL,
                texture_class TEXT,
                organic_carbon_pct REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bbox TEXT NOT NULL,
                health_class TEXT,
                health_prob REAL,
                pest_prob REAL,
                crop_health_score REAL,
                cnn_confidence REAL,
                lstm_confidence REAL,
                risk_level TEXT,
                recommendation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                date TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                predicted_class TEXT,
                actual_class TEXT,
                confidence_rating INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
        
        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                training_date TEXT,
                training_samples INTEGER,
                validation_accuracy REAL,
                f1_score REAL,
                model_path TEXT,
                is_active BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                bbox TEXT,
                crop_type TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT,
                results_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pest risk history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pest_risk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                location_hash TEXT NOT NULL,
                pest_risk REAL NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                ndvi REAL,
                crop_type TEXT,
                risk_factors TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_satellite_date ON satellite_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_pred ON user_feedback(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pest_location ON pest_risk_history(location_hash, date)")
        
        self.conn.commit()
        logger.info("Database schema initialized successfully")
    
    def insert_satellite_data(self, data: dict):
        """Insert satellite data record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO satellite_data 
            (date, bbox, ndvi_mean, ndvi_std, ndwi_mean, ndwi_std, 
             evi_mean, evi_std, cloud_cover, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('date'),
            str(data.get('bbox')),
            data.get('ndvi_mean'),
            data.get('ndvi_std'),
            data.get('ndwi_mean'),
            data.get('ndwi_std'),
            data.get('evi_mean'),
            data.get('evi_std'),
            data.get('cloud_cover'),
            data.get('image_path')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_weather_data(self, data: dict):
        """Insert weather data record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO weather_data 
            (date, latitude, longitude, temperature_c, humidity_pct, 
             rainfall_mm, windspeed_ms, lst_c, cloud_cover_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('date'),
            data.get('latitude'),
            data.get('longitude'),
            data.get('temperature'),
            data.get('humidity'),
            data.get('rainfall'),
            data.get('wind_speed'),
            data.get('lst'),
            data.get('cloud_cover')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_prediction(self, data: dict):
        """Insert prediction record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (date, bbox, health_class, health_prob, pest_prob, 
             crop_health_score, cnn_confidence, lstm_confidence, 
             risk_level, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('date'),
            str(data.get('bbox')),
            data.get('health_class'),
            data.get('health_prob'),
            data.get('pest_prob'),
            data.get('crop_health_score'),
            data.get('cnn_confidence'),
            data.get('lstm_confidence'),
            data.get('risk_level'),
            data.get('recommendation')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_predictions(self, bbox: str = None, date: str = None, limit: int = 100):
        """Retrieve predictions"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if bbox:
            query += " AND bbox = ?"
            params.append(bbox)
        
        if date:
            query += " AND date = ?"
            params.append(date)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def insert_pest_risk_history(self, data: dict):
        """Insert pest risk history record"""
        cursor = self.conn.cursor()
        
        # Create location hash for easy querying
        location_hash = f"{data.get('latitude'):.4f}_{data.get('longitude'):.4f}"
        
        cursor.execute("""
            INSERT INTO pest_risk_history 
            (date, latitude, longitude, location_hash, pest_risk, 
             temperature_c, humidity_pct, ndvi, crop_type, risk_factors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('date'),
            data.get('latitude'),
            data.get('longitude'),
            location_hash,
            data.get('pest_risk'),
            data.get('temperature'),
            data.get('humidity'),
            data.get('ndvi'),
            data.get('crop_type'),
            str(data.get('risk_factors', []))
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_pest_risk_history(self, latitude: float, longitude: float, days: int = 90):
        """Retrieve pest risk history for a location"""
        cursor = self.conn.cursor()
        
        location_hash = f"{latitude:.4f}_{longitude:.4f}"
        
        cursor.execute("""
            SELECT date, pest_risk, temperature_c, humidity_pct, ndvi, 
                   crop_type, risk_factors, created_at
            FROM pest_risk_history
            WHERE location_hash = ?
            ORDER BY date DESC
            LIMIT ?
        """, (location_hash, days))
        
        return cursor.fetchall()


def initialize_database():
    """Initialize database with schema"""
    db = DatabaseManager()
    db.connect()
    db.initialize_schema()
    
    # Insert sample model version
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO model_versions 
        (model_name, version, training_date, model_path, is_active)
        VALUES 
        ('MobileNetV2_CropHealth', '1.0.0', ?, 'models/trained/mobilenetv2_crop_health.h5', 1),
        ('LSTM_PestRisk', '1.0.0', ?, 'models/trained/lstm_pest_risk.h5', 1)
    """, (datetime.now().isoformat(), datetime.now().isoformat()))
    
    db.conn.commit()
    db.close()
    
    logger.info("Database initialized successfully!")


if __name__ == "__main__":
    initialize_database()
