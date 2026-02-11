"""
Web Application Launcher
Launches the Flask web application with the trained model
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from cnn_model import DigitRecognitionCNN
from data_preprocessing import DataPreprocessor
from database import RecognitionDatabase
from web_app import app, initialize_app, create_html_templates
import sys

def main():
    """Launch web application"""
    print("="*70)
    print("  HANDWRITTEN DIGIT RECOGNITION - WEB APPLICATION")
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
    database = RecognitionDatabase(db_path='database/web_recognition.db')
    print("✓ Database initialized")
    
    # Create HTML templates
    print("\nCreating HTML templates...")
    create_html_templates()
    print("✓ Templates created")
    
    # Initialize Flask app
    print("\nInitializing Flask application...")
    initialize_app(cnn, preprocessor, database)
    print("✓ Flask app initialized")
    
    # Run server
    print("\n" + "="*70)
    print("  SERVER STARTING")
    print("="*70)
    print("\n✓ Web server is running!")
    print("\n📱 Open your browser and navigate to:")
    print("   http://localhost:5000")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*70)
    print()
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        database.close()
        print("✓ Server stopped successfully")
    except Exception as e:
        print(f"\nError running server: {e}")
        import traceback
        traceback.print_exc()
        database.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
