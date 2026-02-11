"""
Desktop Application Launcher
Launches the Tkinter GUI application with the trained model
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from cnn_model import DigitRecognitionCNN
from data_preprocessing import DataPreprocessor
from database import RecognitionDatabase
from desktop_gui import DigitRecognitionGUI
import sys

def main():
    """Launch desktop application"""
    print("="*70)
    print("  HANDWRITTEN DIGIT RECOGNITION - DESKTOP APPLICATION")
    print("="*70)
    print()
    
    # Check if model exists
    model_path = 'models/digit_recognition_model.h5'
    if not os.path.exists(model_path):
        print("ERROR: Trained model not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease train the model first by running:")
        print("  python train_model.py")
        sys.exit(1)
    
    # Load model
    print("Loading trained model...")
    cnn = DigitRecognitionCNN()
    cnn.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = DataPreprocessor()
    print("✓ Preprocessor initialized")
    
    # Initialize database
    print("\nInitializing database...")
    database = RecognitionDatabase(db_path='database/desktop_recognition.db')
    print("✓ Database initialized")
    
    # Launch GUI
    print("\nLaunching desktop application...")
    print("="*70)
    print()
    
    try:
        app = DigitRecognitionGUI(cnn, preprocessor, database)
        app.run()
    except Exception as e:
        print(f"\nError launching application: {e}")
        import traceback
        traceback.print_exc()
        database.close()
        sys.exit(1)
    
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
