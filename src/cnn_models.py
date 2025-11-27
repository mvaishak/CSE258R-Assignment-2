"""
CNN Models Module for DiamondHood
Image classification and embedding generation for visual similarity
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
from PIL import Image
from pathlib import Path
import joblib
import json


class DiamondCNNClassifier:
    """
    CNN for diamond quality classification from images.
    """
    
    def __init__(self, image_size=(224, 224), num_classes=3):
        """
        Initialize CNN classifier.
        
        Args:
            image_size: Target image dimensions
            num_classes: Number of quality classes to predict
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_custom_cnn(self):
        """
        Build a custom CNN architecture.
        
        Returns:
            Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.image_size, 3)),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model='efficientnet', trainable_layers=10):
        """
        Build transfer learning model using pre-trained networks.
        
        Args:
            base_model: 'resnet', 'efficientnet', or 'vgg'
            trainable_layers: Number of top layers to fine-tune
            
        Returns:
            Keras model
        """
        # Load base model
        if base_model == 'resnet':
            base = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        elif base_model == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        else:  # vgg
            base = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        
        # Freeze base layers except last N
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Build full model
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        print("✓ Model compiled successfully")
        print(f"Total parameters: {model.count_params():,}")
        
    def get_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators for training and validation.
        
        Args:
            train_dir: Training images directory
            val_dir: Validation images directory
            batch_size: Batch size
            
        Returns:
            train_generator, val_generator
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50, 
              models_dir='models', model_name='diamond_cnn'):
        """
        Train the CNN model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            models_dir: Directory to save models
            model_name: Name for saving model
            
        Returns:
            Training history
        """
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(models_path / f'{model_name}_best.h5'),
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
        
        # Train
        print(f"\nStarting training for {epochs} epochs...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        return self.history
    
    def evaluate(self, test_generator):
        """
        Evaluate model on test data.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating model...")
        results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_k_accuracy': results[2]
        }
        
        print(f"\nTest Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Top-K Accuracy: {metrics['top_k_accuracy']*100:.2f}%")
        
        return metrics
    
    def predict_image(self, image_path):
        """
        Predict quality class for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted class probabilities
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        
        return predictions[0]
    
    def plot_training_history(self):
        """
        Plot training and validation metrics.
        """
        if self.history is None:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class DiamondEmbeddingGenerator:
    """
    Generate embeddings for visual similarity search.
    """
    
    def __init__(self, base_model='efficientnet', image_size=(224, 224)):
        """
        Initialize embedding generator.
        
        Args:
            base_model: Pre-trained model to use
            image_size: Target image size
        """
        self.image_size = image_size
        self.model = self._build_embedding_model(base_model)
        
    def _build_embedding_model(self, base_model_name):
        """
        Build model for generating embeddings.
        """
        if base_model_name == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3),
                pooling='avg'
            )
        else:
            base = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3),
                pooling='avg'
            )
        
        return base
    
    def generate_embedding(self, image_path):
        """
        Generate embedding for a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Embedding vector
        """
        # Load and preprocess
        img = Image.open(image_path)
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate embedding
        embedding = self.model.predict(img_array, verbose=0)
        
        return embedding[0]
    
    def generate_batch_embeddings(self, image_paths, batch_size=32):
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        from tqdm import tqdm
        
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating embeddings"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path)
                    img = img.resize(self.image_size)
                    img_array = np.array(img) / 255.0
                    batch_images.append(img_array)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if batch_images:
                batch_array = np.array(batch_images)
                batch_embeddings = self.model.predict(batch_array, verbose=0)
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def save_embeddings(self, embeddings, save_path):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Array of embeddings
            save_path: Path to save file
        """
        np.save(save_path, embeddings)
        print(f"✓ Embeddings saved: {save_path}")
    
    def load_embeddings(self, load_path):
        """
        Load embeddings from disk.
        
        Args:
            load_path: Path to load from
            
        Returns:
            Array of embeddings
        """
        embeddings = np.load(load_path)
        print(f"✓ Embeddings loaded: {load_path}")
        return embeddings


if __name__ == "__main__":
    print("DiamondCNN module loaded successfully")
    print("Available classes:")
    print("  - DiamondCNNClassifier: For quality classification")
    print("  - DiamondEmbeddingGenerator: For visual embeddings")
