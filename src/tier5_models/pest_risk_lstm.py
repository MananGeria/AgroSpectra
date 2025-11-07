"""
Tier 5: AI Modeling Layer - LSTM Pest Risk Predictor
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import yaml
import joblib


class PestRiskLSTM:
    """LSTM-based pest outbreak risk predictor"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['models']['pest_risk_lstm']
        
        self.sequence_length = self.config['sequence_length']
        self.input_features = self.config['input_features']
        self.n_features = len(self.input_features)
        
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self):
        """Build LSTM model"""
        logger.info("Building LSTM pest risk predictor")
        
        model = Sequential(name='pest_risk_lstm')
        
        # Add LSTM layers from config
        lstm_layers = self.config['layers']
        
        for i, layer_config in enumerate(lstm_layers):
            if layer_config.get('activation') == 'sigmoid':
                # Output layer
                model.add(Dense(
                    layer_config['units'],
                    activation=layer_config['activation']
                ))
            else:
                # LSTM layer
                model.add(LSTM(
                    layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False),
                    input_shape=(self.sequence_length, self.n_features) if i == 0 else None
                ))
                
                if 'dropout' in layer_config:
                    model.add(Dropout(layer_config['dropout']))
        
        # Compile model
        train_config = self.config['training']
        model.compile(
            optimizer=Adam(learning_rate=train_config['learning_rate']),
            loss=train_config['loss'],
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        logger.info(f"Model built: {self.model.summary()}")
        return self.model
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        labels: pd.Series = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for LSTM
        
        Args:
            data: DataFrame with time series features
            labels: Series with pest outbreak labels (optional)
            
        Returns:
            Tuple of (sequences, labels) or just sequences if no labels
        """
        # Ensure features are in correct order
        feature_data = data[self.input_features].values
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X = []
        y = [] if labels is not None else None
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            
            if labels is not None:
                y.append(labels.iloc[i + self.sequence_length])
        
        X = np.array(X)
        
        if labels is not None:
            y = np.array(y)
            return X, y
        
        return X, None
    
    def train(
        self,
        data_file: str,
        labels_file: str,
        model_save_path: str = "models/trained/lstm_pest_risk.h5",
        validation_split: float = 0.2
    ):
        """
        Train the LSTM model
        
        Args:
            data_file: Path to time series data CSV
            labels_file: Path to labels CSV
            model_save_path: Path to save trained model
            validation_split: Validation split ratio
        """
        logger.info("Starting LSTM training")
        
        if self.model is None:
            self.build_model()
        
        # Load data
        data = pd.read_csv(data_file)
        labels = pd.read_csv(labels_file)['pest_outbreak']
        
        # Prepare sequences
        X, y = self.prepare_sequences(data, labels)
        
        logger.info(f"Prepared {len(X)} sequences")
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        train_config = self.config['training']
        history = self.model.fit(
            X_train, y_train,
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scaler
        scaler_path = model_save_path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Training complete. Model saved to {model_save_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
        return history
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning("Scaler not found, using new scaler")
        
        return self.model
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict pest outbreak probability for single sequence
        
        Args:
            sequence: Time series sequence of shape (sequence_length, n_features)
            
        Returns:
            Pest outbreak probability (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Scale sequence
        scaled_sequence = self.scaler.transform(sequence)
        
        # Add batch dimension
        scaled_sequence = np.expand_dims(scaled_sequence, axis=0)
        
        # Predict
        probability = self.model.predict(scaled_sequence, verbose=0)[0][0]
        
        return float(probability)
    
    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict pest outbreak probabilities for batch of sequences"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Scale sequences
        scaled_sequences = []
        for seq in sequences:
            scaled_seq = self.scaler.transform(seq)
            scaled_sequences.append(scaled_seq)
        
        scaled_sequences = np.array(scaled_sequences)
        
        # Predict
        probabilities = self.model.predict(scaled_sequences, verbose=0)
        
        return probabilities.flatten()
    
    def predict_from_dataframe(self, data: pd.DataFrame) -> List[float]:
        """
        Predict pest risk from DataFrame of recent observations
        
        Args:
            data: DataFrame with recent time series data
            
        Returns:
            List of pest outbreak probabilities
        """
        X, _ = self.prepare_sequences(data)
        
        if len(X) == 0:
            logger.warning("Not enough data to create sequences")
            return []
        
        return self.predict_batch(X).tolist()
    
    def evaluate(self, test_data_file: str, test_labels_file: str):
        """Evaluate model on test set"""
        logger.info("Evaluating LSTM model")
        
        # Load test data
        data = pd.read_csv(test_data_file)
        labels = pd.read_csv(test_labels_file)['pest_outbreak']
        
        # Prepare sequences
        X_test, y_test = self.prepare_sequences(data, labels)
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info(f"Test metrics: {metrics}")
        
        # Calculate additional metrics
        y_pred = self.model.predict(X_test, verbose=0)
        
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score
        )
        
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred_binary))
        
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred_binary))
        
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"\nROC AUC: {auc:.4f}")
        
        return metrics


def create_training_sequences(
    satellite_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    pest_records: pd.DataFrame,
    output_dir: str
):
    """
    Create training sequences from satellite, weather, and pest data
    
    Args:
        satellite_data: DataFrame with NDVI and other indices
        weather_data: DataFrame with weather parameters
        pest_records: DataFrame with pest outbreak records
        output_dir: Output directory for training files
    """
    logger.info("Creating training sequences")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Merge datasets by date
    merged = satellite_data.merge(weather_data, on='date', how='inner')
    merged = merged.merge(pest_records, on='date', how='left')
    
    # Fill missing pest records with 0 (no outbreak)
    merged['pest_outbreak'] = merged['pest_outbreak'].fillna(0).astype(int)
    
    # Select relevant features
    features = ['ndvi', 'temperature', 'humidity', 'rainfall', 'lst']
    feature_data = merged[features]
    labels = merged['pest_outbreak']
    
    # Save to CSV
    feature_data.to_csv(output_path / 'sequences.csv', index=False)
    labels.to_csv(output_path / 'labels.csv', index=False)
    
    logger.info(f"Training sequences saved to {output_path}")
    
    return feature_data, labels


if __name__ == "__main__":
    # Example: Build and summarize model
    lstm = PestRiskLSTM()
    lstm.build_model()
    
    print("\nModel architecture:")
    print(lstm.model.summary())
    
    # Example: Train model (requires data)
    # lstm.train(
    #     data_file="data/training/pest_sequences/sequences.csv",
    #     labels_file="data/training/pest_sequences/labels.csv"
    # )
