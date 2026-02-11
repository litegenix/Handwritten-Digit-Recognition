"""
Flask Web Application for Digit Recognition
Provides web interface for uploading images and recognizing digits
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import cv2
import json
import secrets

# Flask app setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'web_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model, preprocessor, and database
model = None
preprocessor = None
database = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_app(cnn_model, data_preprocessor, db):
    """
    Initialize Flask app with model and database
    
    Args:
        cnn_model: Trained CNN model
        data_preprocessor: Data preprocessor instance
        db: Database instance
    """
    global model, preprocessor, database
    model = cnn_model
    preprocessor = data_preprocessor
    database = db
    print("Flask app initialized with model and database")

@app.route('/')
def index():
    """Home page"""
    # Create session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        session['prediction_count'] = 0
        if database:
            database.create_session(session['session_id'], interface_type='web')
    
    return render_template('index.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or BMP'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        start_time = datetime.now()
        preprocessed = preprocessor.preprocess_image(filepath)
        
        # Get prediction
        predicted_digit, confidence = model.predict(preprocessed)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log to database
        if database:
            database.log_prediction(
                predicted_digit=int(predicted_digit),
                confidence_scores=confidence,
                image_source='web_upload',
                processing_time=processing_time
            )
        
        # Update session counter
        session['prediction_count'] = session.get('prediction_count', 0) + 1
        
        # Convert preprocessed image to base64 for display
        img_display = (preprocessed.squeeze() * 255).astype(np.uint8)
        img_display = cv2.resize(img_display, (140, 140), interpolation=cv2.INTER_NEAREST)
        
        pil_img = Image.fromarray(img_display)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare response
        response = {
            'success': True,
            'predicted_digit': int(predicted_digit),
            'confidence': float(confidence[predicted_digit]) * 100,
            'all_probabilities': (confidence * 100).tolist(),
            'processing_time': processing_time,
            'preprocessed_image': f"data:image/png;base64,{img_str}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_drawn', methods=['POST'])
def predict_drawn():
    """Handle drawn digit prediction"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Preprocess
        start_time = datetime.now()
        preprocessed = preprocessor.preprocess_drawn_digit(img)
        
        # Predict
        predicted_digit, confidence = model.predict(preprocessed)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log to database
        if database:
            database.log_prediction(
                predicted_digit=int(predicted_digit),
                confidence_scores=confidence,
                image_source='web_drawn',
                processing_time=processing_time
            )
        
        # Update session counter
        session['prediction_count'] = session.get('prediction_count', 0) + 1
        
        # Convert preprocessed image to base64
        img_display = (preprocessed.squeeze() * 255).astype(np.uint8)
        img_display = cv2.resize(img_display, (140, 140), interpolation=cv2.INTER_NEAREST)
        
        pil_img = Image.fromarray(img_display)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Response
        response = {
            'success': True,
            'predicted_digit': int(predicted_digit),
            'confidence': float(confidence[predicted_digit]) * 100,
            'all_probabilities': (confidence * 100).tolist(),
            'processing_time': processing_time,
            'preprocessed_image': f"data:image/png;base64,{img_str}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def statistics():
    """Get prediction statistics"""
    try:
        if database:
            stats = database.get_statistics(days=7)
            session_count = session.get('prediction_count', 0)
            
            return jsonify({
                'success': True,
                'session_predictions': session_count,
                'total_predictions_7days': stats['total_predictions'],
                'average_confidence': stats['average_confidence'] * 100,
                'predictions_by_digit': stats['predictions_by_digit'],
                'predictions_by_source': stats['predictions_by_source']
            })
        else:
            return jsonify({'success': False, 'error': 'Database not available'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/recent_predictions')
def recent_predictions():
    """Get recent predictions"""
    try:
        if database:
            recent = database.get_recent_predictions(limit=10)
            return jsonify({
                'success': True,
                'predictions': recent
            })
        else:
            return jsonify({'success': False, 'error': 'Database not available'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def create_html_templates():
    """Create HTML templates for Flask app"""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
        }
        
        .panel h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        #drawCanvas {
            border: 3px solid #2c3e50;
            border-radius: 10px;
            cursor: crosshair;
            background: black;
            display: block;
            margin: 0 auto;
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #27ae60;
            color: white;
        }
        
        .btn-primary:hover {
            background: #229954;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(39,174,96,0.3);
        }
        
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }
        
        .btn-info {
            background: #3498db;
            color: white;
        }
        
        .btn-info:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-top: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: #ebf5fb;
            border-color: #2980b9;
        }
        
        .upload-area input {
            display: none;
        }
        
        .results {
            text-align: center;
            margin-top: 30px;
        }
        
        .digit-display {
            font-size: 5em;
            font-weight: bold;
            color: #27ae60;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .confidence {
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        
        .preview-box {
            border: 2px solid #34495e;
            border-radius: 10px;
            padding: 10px;
            background: white;
            margin: 20px auto;
            width: fit-content;
        }
        
        #preprocessedImage {
            border-radius: 5px;
        }
        
        .prob-bars {
            margin-top: 20px;
        }
        
        .prob-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 10px;
        }
        
        .prob-label {
            font-weight: bold;
            width: 30px;
        }
        
        .prob-bar-container {
            flex: 1;
            background: #ecf0f1;
            border-radius: 10px;
            height: 24px;
            overflow: hidden;
        }
        
        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            border-radius: 10px;
            transition: width 0.5s;
        }
        
        .prob-value {
            font-size: 0.9em;
            width: 50px;
            text-align: right;
        }
        
        .slider-container {
            margin-top: 15px;
        }
        
        .slider-container label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        input[type="range"] {
            width: 100%;
        }
        
        .stats {
            background: #e8f8f5;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .stats h3 {
            color: #16a085;
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✍️ Handwritten Digit Recognition</h1>
            <p>Draw or upload a digit (0-9) for instant recognition</p>
        </div>
        
        <div class="main-content">
            <!-- Left Panel: Drawing Area -->
            <div class="panel">
                <h2>Draw a Digit</h2>
                <canvas id="drawCanvas" width="280" height="280"></canvas>
                
                <div class="slider-container">
                    <label for="brushSize">Brush Size: <span id="brushValue">15</span></label>
                    <input type="range" id="brushSize" min="5" max="30" value="15">
                </div>
                
                <div class="controls">
                    <button class="btn-primary" onclick="recognizeDrawn()">🔍 Recognize</button>
                    <button class="btn-danger" onclick="clearCanvas()">🗑️ Clear</button>
                </div>
                
                <h2 style="margin-top: 30px;">Upload an Image</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <p style="font-size: 3em;">📁</p>
                    <p>Click to upload an image</p>
                    <p style="font-size: 0.9em; color: #7f8c8d;">PNG, JPG, or BMP</p>
                    <input type="file" id="fileInput" accept="image/*" onchange="uploadImage()">
                </div>
            </div>
            
            <!-- Right Panel: Results -->
            <div class="panel">
                <h2>Recognition Results</h2>
                
                <div class="preview-box">
                    <p style="margin-bottom: 10px; font-weight: bold;">Preprocessed (28x28):</p>
                    <img id="preprocessedImage" width="140" height="140" style="display:none;">
                    <div id="previewPlaceholder" style="width:140px; height:140px; background:#ecf0f1; border-radius:5px;"></div>
                </div>
                
                <div class="results">
                    <div class="digit-display" id="predictedDigit">?</div>
                    <div class="confidence" id="confidenceText">Confidence: ---%</div>
                </div>
                
                <div class="prob-bars" id="probabilityBars"></div>
                
                <div class="stats" id="sessionStats">
                    <h3>Session Statistics</h3>
                    <p id="statsText">Predictions: 0</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Canvas setup
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        let brushSize = 15;
        let sessionPredictions = 0;
        
        // Initialize canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        
        // Drawing functions
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            ctx.lineWidth = brushSize;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            
            [lastX, lastY] = [e.offsetX, e.offsetY];
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            
            if (e.type === 'touchstart') {
                isDrawing = true;
                [lastX, lastY] = [x, y];
            } else if (e.type === 'touchmove' && isDrawing) {
                ctx.lineWidth = brushSize;
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.stroke();
                [lastX, lastY] = [x, y];
            }
        }
        
        // Brush size control
        document.getElementById('brushSize').addEventListener('input', function(e) {
            brushSize = e.target.value;
            document.getElementById('brushValue').textContent = brushSize;
        });
        
        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            resetResults();
        }
        
        function resetResults() {
            document.getElementById('predictedDigit').textContent = '?';
            document.getElementById('predictedDigit').style.color = '#2c3e50';
            document.getElementById('confidenceText').textContent = 'Confidence: ---%';
            document.getElementById('probabilityBars').innerHTML = '';
            document.getElementById('preprocessedImage').style.display = 'none';
            document.getElementById('previewPlaceholder').style.display = 'block';
        }
        
        async function recognizeDrawn() {
            const imageData = canvas.toDataURL('image/png');
            
            try {
                const response = await fetch('/predict_drawn', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: imageData })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result);
                    sessionPredictions++;
                    updateStats();
                } else {
                    alert('Error: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
        
        async function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict_upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result);
                    sessionPredictions++;
                    updateStats();
                } else {
                    alert('Error: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
        
        function displayResults(result) {
            // Show predicted digit
            document.getElementById('predictedDigit').textContent = result.predicted_digit;
            document.getElementById('predictedDigit').style.color = '#27ae60';
            
            // Show confidence
            const confidence = result.confidence.toFixed(1);
            document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
            document.getElementById('confidenceText').style.color = 
                confidence > 80 ? '#27ae60' : '#f39c12';
            
            // Show preprocessed image
            const imgElement = document.getElementById('preprocessedImage');
            imgElement.src = result.preprocessed_image;
            imgElement.style.display = 'block';
            document.getElementById('previewPlaceholder').style.display = 'none';
            
            // Show probability bars
            displayProbabilityBars(result.all_probabilities, result.predicted_digit);
        }
        
        function displayProbabilityBars(probabilities, predictedDigit) {
            const container = document.getElementById('probabilityBars');
            container.innerHTML = '';
            
            for (let i = 0; i < 10; i++) {
                const prob = probabilities[i];
                const isHighest = i === predictedDigit;
                
                const item = document.createElement('div');
                item.className = 'prob-item';
                
                item.innerHTML = `
                    <div class="prob-label">${i}:</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width: ${prob}%; background: ${isHighest ? '#27ae60' : '#3498db'};"></div>
                    </div>
                    <div class="prob-value">${prob.toFixed(1)}%</div>
                `;
                
                container.appendChild(item);
            }
        }
        
        function updateStats() {
            document.getElementById('statsText').textContent = `Predictions: ${sessionPredictions}`;
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    print("HTML templates created successfully!")

if __name__ == "__main__":
    print("Flask Web Application")
    print("\nThis module requires the trained model to run.")
    print("Please run train_model.py first, then:")
    print("  python web_app.py")
    
    # Create templates
    create_html_templates()
