"""
Tier 5: AI Modeling Layer - MobileNetV2 Crop Health Classifier
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loguru import logger
import yaml


class CropHealthCNN:
    """MobileNetV2-based crop health classifier"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['models']['crop_health_cnn']
        
        self.input_shape = tuple(self.config['input_shape'])
        self.num_classes = len(self.config['classes'])
        self.class_names = self.config['classes']
        
        self.model = None
    
    def build_model(self):
        """Build MobileNetV2 model with custom classification head"""
        logger.info("Building MobileNetV2 crop health classifier")
        
        # Load pretrained MobileNetV2 (without top layer)
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build custom classification head
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs, outputs, name='crop_health_cnn')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
            loss=self.config['training']['loss'],
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"Model built: {self.model.summary()}")
        return self.model
    
    def create_data_generators(self, train_dir: str, val_dir: str):
        """Create data generators with augmentation"""
        train_config = self.config['training']
        aug_config = self.config['augmentation']
        
        # Training data generator with augmentation
        if aug_config['enabled']:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=aug_config['rotation_range'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                brightness_range=aug_config['brightness_range'],
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=train_config['batch_size'],
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=train_config['batch_size'],
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(
        self,
        train_dir: str,
        val_dir: str,
        model_save_path: str = "models/trained/mobilenetv2_crop_health.h5"
    ):
        """Train the model"""
        logger.info("Starting model training")
        
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_dir, val_dir)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        train_config = self.config['training']
        history = self.model.fit(
            train_gen,
            epochs=train_config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training complete. Model saved to {model_save_path}")
        return history
    
    def fine_tune(self, train_gen, val_gen, epochs: int = 20):
        """Fine-tune the model by unfreezing base layers"""
        logger.info("Fine-tuning model")
        
        # Unfreeze base model
        self.model.layers[1].trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=self.config['training']['loss'],
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1
        )
        
        return history
    
    def load_model(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        return self.model
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict crop health for single image
        
        Args:
            image: Image array of shape (height, width, 3)
            
        Returns:
            Dictionary with class probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        if image.shape != self.input_shape:
            image = tf.image.resize(image, self.input_shape[:2])
        
        image = np.expand_dims(image, axis=0) / 255.0
        
        # Predict
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Create result dictionary
        result = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, predictions)
        }
        
        result['predicted_class'] = self.class_names[np.argmax(predictions)]
        result['confidence'] = float(np.max(predictions))
        
        return result
    
    def predict_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Predict crop health for batch of images"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess images
        if images.shape[1:] != self.input_shape:
            images = tf.image.resize(images, self.input_shape[:2])
        
        images = images / 255.0
        
        # Predict
        predictions = self.model.predict(images, verbose=0)
        
        # Create results
        results = []
        for pred in predictions:
            result = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, pred)
            }
            result['predicted_class'] = self.class_names[np.argmax(pred)]
            result['confidence'] = float(np.max(pred))
            results.append(result)
        
        return results
    
    def evaluate(self, test_dir: str):
        """Evaluate model on test set"""
        logger.info("Evaluating model")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=self.config['training']['batch_size'],
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        # Evaluate
        results = self.model.evaluate(test_gen, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info(f"Test metrics: {metrics}")
        
        return metrics


def create_training_tiles(
    ndvi_path: str,
    ndwi_path: str,
    evi_path: str,
    output_dir: str,
    tile_size: int = 128,
    stride: int = 64
):
    """
    Create training tiles from vegetation index rasters
    
    Args:
        ndvi_path: Path to NDVI raster
        ndwi_path: Path to NDWI raster
        evi_path: Path to EVI raster
        output_dir: Output directory for tiles
        tile_size: Size of tiles
        stride: Stride for sliding window
    """
    import rasterio
    
    logger.info("Creating training tiles")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read rasters
    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi = src_ndvi.read(1)
    
    with rasterio.open(ndwi_path) as src_ndwi:
        ndwi = src_ndwi.read(1)
    
    with rasterio.open(evi_path) as src_evi:
        evi = src_evi.read(1)
    
    # Stack into 3-channel image
    stacked = np.dstack([ndvi, ndwi, evi])
    
    # Normalize to 0-255
    stacked = ((stacked + 1) / 2 * 255).astype(np.uint8)
    
    # Extract tiles
    tiles = []
    height, width = ndvi.shape
    
    for i in range(0, height - tile_size + 1, stride):
        for j in range(0, width - tile_size + 1, stride):
            tile = stacked[i:i+tile_size, j:j+tile_size, :]
            tiles.append(tile)
    
    logger.info(f"Created {len(tiles)} tiles")
    
    return np.array(tiles)


if __name__ == "__main__":
    # Example: Build and summarize model
    cnn = CropHealthCNN()
    cnn.build_model()
    
    print("\nModel architecture:")
    print(cnn.model.summary())
    
    # Example: Train model (requires data)
    # cnn.train(
    #     train_dir="data/training/crop_health/train",
    #     val_dir="data/training/crop_health/val"
    # )
