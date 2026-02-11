"""
CNN Model Architecture and Training Module
Implements a Convolutional Neural Network for handwritten digit recognition
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

class DigitRecognitionCNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize CNN model for digit recognition
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of output classes (digits 0-9)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, architecture='advanced'):
        """
        Build CNN architecture
        
        Args:
            architecture (str): 'simple', 'standard', or 'advanced'
        """
        if architecture == 'simple':
            self.model = self._build_simple_model()
        elif architecture == 'standard':
            self.model = self._build_standard_model()
        else:
            self.model = self._build_advanced_model()
        
        return self.model
    
    def _build_simple_model(self):
        """Build simple CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_standard_model(self):
        """Build standard CNN model with dropout"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_advanced_model(self):
        """Build advanced CNN model with residual connections"""
        inputs = layers.Input(shape=self.input_shape)
        
        # First Conv Block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second Conv Block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third Conv Block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Dense Layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("Model compiled successfully!")
        print(f"\nModel Summary:")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=128, use_augmentation=True):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            use_augmentation (bool): Whether to use data augmentation
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
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
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Data augmentation
        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                shear_range=0.1
            )
            datagen.fit(X_train)
            
            print("\n=== Training with Data Augmentation ===")
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("\n=== Training without Data Augmentation ===")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
        """
        print("\n=== Model Evaluation ===")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # F1 Score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print(f"Test F1-Score: {f1_score:.4f}")
        
        # Classification report
        print("\n=== Classification Report ===")
        print(classification_report(y_true_classes, y_pred_classes, 
                                   target_names=[str(i) for i in range(10)]))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'loss': test_loss,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'true_labels': y_true_classes
        }
    
    def plot_training_history(self, save_path='models/training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path='models/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def save_model(self, filepath='models/digit_recognition_model.h5'):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
        
        # Save model info
        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = filepath.replace('.h5', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    
    def load_model(self, filepath='models/digit_recognition_model.h5'):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"\nModel loaded from: {filepath}")
        return self.model
    
    def predict(self, image):
        """
        Predict digit from image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            tuple: (predicted_digit, confidence_scores)
        """
        # Ensure correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0]
        
        return predicted_digit, confidence


if __name__ == "__main__":
    print("=== Testing CNN Model Module ===\n")
    
    # Create model instance
    print("1. Creating CNN model...")
    cnn = DigitRecognitionCNN()
    
    # Build model
    print("\n2. Building advanced architecture...")
    model = cnn.build_model(architecture='advanced')
    
    # Compile model
    print("\n3. Compiling model...")
    cnn.compile_model(learning_rate=0.001)
    
    # Create dummy data for testing
    print("\n4. Creating dummy data for testing...")
    X_dummy = np.random.rand(100, 28, 28, 1)
    y_dummy = keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
    
    print("\n5. Testing prediction...")
    test_image = np.random.rand(28, 28, 1)
    digit, confidence = cnn.predict(test_image)
    print(f"   Predicted digit: {digit}")
    print(f"   Confidence scores: {confidence}")
    
    print("\n✓ CNN Model module tested successfully!")
