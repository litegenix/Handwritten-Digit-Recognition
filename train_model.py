"""
Model Training Script
Trains the CNN model on MNIST dataset and saves it
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from cnn_model import DigitRecognitionCNN
from data_preprocessing import DataPreprocessor
import argparse
import json
from datetime import datetime

def train_model(architecture='advanced', epochs=30, batch_size=128, 
                use_augmentation=True, learning_rate=0.001):
    """
    Train digit recognition model
    
    Args:
        architecture (str): Model architecture ('simple', 'standard', 'advanced')
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        use_augmentation (bool): Whether to use data augmentation
        learning_rate (float): Learning rate
    """
    print("="*70)
    print("  HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
    print("="*70)
    print()
    
    # Initialize preprocessor
    print("Step 1: Loading and preprocessing MNIST data...")
    preprocessor = DataPreprocessor()
    
    # Load MNIST
    preprocessor.load_mnist_data()
    
    # Preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.preprocess_data(
        validation_split=0.1
    )
    
    # Create visualizations
    print("\nCreating data visualizations...")
    preprocessor.visualize_samples(num_samples=10)
    preprocessor.visualize_class_distribution()
    
    # Get statistics
    stats = preprocessor.get_data_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    print(f"\nStep 2: Building {architecture} CNN architecture...")
    cnn = DigitRecognitionCNN(input_shape=(28, 28, 1), num_classes=10)
    cnn.build_model(architecture=architecture)
    
    # Compile model
    print("\nStep 3: Compiling model...")
    cnn.compile_model(learning_rate=learning_rate)
    
    # Train model
    print(f"\nStep 4: Training model...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data augmentation: {use_augmentation}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    history = cnn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_augmentation=use_augmentation
    )
    
    # Evaluate model
    print("\nStep 5: Evaluating model on test set...")
    eval_results = cnn.evaluate(X_test, y_test)
    
    # Plot training history
    print("\nStep 6: Generating visualizations...")
    cnn.plot_training_history()
    cnn.plot_confusion_matrix(eval_results['confusion_matrix'])
    
    # Save model
    print("\nStep 7: Saving model...")
    model_path = 'models/digit_recognition_model.h5'
    cnn.save_model(model_path)
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'architecture': architecture,
        'epochs': epochs,
        'batch_size': batch_size,
        'use_augmentation': use_augmentation,
        'learning_rate': learning_rate,
        'final_accuracy': float(eval_results['accuracy']),
        'final_precision': float(eval_results['precision']),
        'final_recall': float(eval_results['recall']),
        'final_f1_score': float(eval_results['f1_score']),
        'final_loss': float(eval_results['loss'])
    }
    
    with open('models/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=4)
    
    # Print summary
    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall:    {eval_results['recall']:.4f}")
    print(f"  F1-Score:  {eval_results['f1_score']:.4f}")
    print(f"  Loss:      {eval_results['loss']:.4f}")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Training history: models/training_history.png")
    print(f"Confusion matrix: models/confusion_matrix.png")
    print(f"Training info: models/training_info.json")
    
    print("\n" + "="*70)
    print("  You can now run the desktop or web application!")
    print("="*70)
    print("\nDesktop App: python desktop_app.py")
    print("Web App:     python web_app_launcher.py")
    
    return cnn, eval_results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train handwritten digit recognition model')
    
    parser.add_argument('--architecture', type=str, default='advanced',
                       choices=['simple', 'standard', 'advanced'],
                       help='CNN architecture to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
