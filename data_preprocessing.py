"""
Data Loading and Preprocessing Module
Handles MNIST dataset loading, preprocessing, and augmentation
"""

import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
    def load_mnist_data(self):
        """
        Load MNIST dataset
        
        Returns:
            tuple: Training and test data
        """
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Image shape: {X_train.shape[1:]}") 
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        return (X_train, y_train), (X_test, y_test)
    
    def preprocess_data(self, validation_split=0.1):
        """
        Preprocess and normalize data
        
        Args:
            validation_split (float): Fraction of training data for validation
            
        Returns:
            tuple: Preprocessed training, validation, and test data
        """
        print("\nPreprocessing data...")
        
        # Normalize pixel values to [0, 1]
        X_train = self.X_train.astype('float32') / 255.0
        X_test = self.X_test.astype('float32') / 255.0
        
        # Reshape to include channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, self.y_train,
            test_size=validation_split,
            random_state=42,
            stratify=self.y_train
        )
        
        # Convert labels to categorical (one-hot encoding)
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_val_cat = keras.utils.to_categorical(y_val, 10)
        y_test_cat = keras.utils.to_categorical(self.y_test, 10)
        
        self.X_train = X_train
        self.y_train = y_train_cat
        self.X_val = X_val
        self.y_val = y_val_cat
        self.X_test = X_test
        self.y_test = y_test_cat
        
        print(f"\nData shapes after preprocessing:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)
    
    def preprocess_image(self, image_path_or_array, target_size=(28, 28)):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path_or_array: Path to image file or numpy array
            target_size (tuple): Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image from {image_path_or_array}")
        else:
            image = image_path_or_array
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        if image.shape != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Invert if needed (MNIST has white digits on black background)
        # Check if background is lighter than foreground
        if np.mean(image) > 127:
            image = 255 - image
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def preprocess_drawn_digit(self, canvas_array, target_size=(28, 28)):
        """
        Preprocess digit drawn on canvas
        
        Args:
            canvas_array: Canvas image array (from drawing app)
            target_size (tuple): Target size
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(canvas_array.shape) == 3:
            gray = cv2.cvtColor(canvas_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = canvas_array
        
        # Find bounding box of digit
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2*padding)
            h = min(gray.shape[0] - y, h + 2*padding)
            
            # Crop to bounding box
            digit = gray[y:y+h, x:x+w]
        else:
            digit = gray
        
        # Make square by padding
        h, w = digit.shape
        if h > w:
            pad = (h - w) // 2
            digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, 
                                      cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            pad = (w - h) // 2
            digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, 
                                      cv2.BORDER_CONSTANT, value=0)
        
        # Resize to target size
        digit = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        digit = digit.astype('float32') / 255.0
        
        # Add channel dimension
        digit = np.expand_dims(digit, axis=-1)
        
        return digit
    
    def visualize_samples(self, num_samples=10, save_path='visualizations/sample_digits.png'):
        """
        Visualize random samples from training data
        
        Args:
            num_samples (int): Number of samples to display
            save_path (str): Path to save visualization
        """
        if self.X_train is None:
            print("No training data loaded!")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Select random samples
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.X_train[idx].squeeze(), cmap='gray')
            
            # Get label
            if len(self.y_train.shape) > 1:  # One-hot encoded
                label = np.argmax(self.y_train[idx])
            else:
                label = self.y_train[idx]
            
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSample visualization saved to: {save_path}")
        plt.close()
    
    def visualize_class_distribution(self, save_path='visualizations/class_distribution.png'):
        """
        Visualize distribution of classes in dataset
        
        Args:
            save_path (str): Path to save visualization
        """
        if self.y_train is None:
            print("No training labels loaded!")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get labels
        if len(self.y_train.shape) > 1:  # One-hot encoded
            labels = np.argmax(self.y_train, axis=1)
        else:
            labels = self.y_train
        
        # Count occurrences
        unique, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Digit', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Digits in Training Set', fontsize=14, fontweight='bold')
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (digit, count) in enumerate(zip(unique, counts)):
            plt.text(digit, count + 50, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved to: {save_path}")
        plt.close()
    
    def get_data_statistics(self):
        """Get statistics about the dataset"""
        stats = {
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'val_samples': len(self.X_val) if self.X_val is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'image_shape': self.X_train.shape[1:] if self.X_train is not None else None,
            'num_classes': 10
        }
        
        return stats


if __name__ == "__main__":
    print("=== Testing Data Preprocessing Module ===\n")
    
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("1. Loading MNIST data...")
    (X_train, y_train), (X_test, y_test) = preprocessor.load_mnist_data()
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.preprocess_data()
    
    # Get statistics
    print("\n3. Dataset statistics:")
    stats = preprocessor.get_data_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Visualize samples
    print("\n4. Creating visualizations...")
    preprocessor.visualize_samples(num_samples=10)
    preprocessor.visualize_class_distribution()
    
    # Test single image preprocessing
    print("\n5. Testing single image preprocessing...")
    test_image = X_train[0].squeeze()
    preprocessed = preprocessor.preprocess_image(test_image)
    print(f"   Original shape: {test_image.shape}")
    print(f"   Preprocessed shape: {preprocessed.shape}")
    print(f"   Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    print("\n✓ Data preprocessing module tested successfully!")
