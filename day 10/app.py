from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

app = Flask(__name__)

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('runs', exist_ok=True)

# Load YOLO model
MODEL_PATH = 'yolov8n.pt'

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html') if os.path.exists('templates/index.html') else '''
    <html>
    <head>
        <title>YOLO Object Detection</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            input[type="file"] { margin: 10px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            img { max-width: 100%; margin: 20px 0; }
            .results { margin-top: 20px; padding: 20px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO Object Detection</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageInput" name="file" accept="image/*" required>
                <button type="submit">Detect Objects</button>
            </form>
            <div id="results"></div>
        </div>
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('imageInput').files[0]);
                
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                let html = '<div class="results"><h2>Results:</h2>';
                if(data.success) {
                    html += `<p>Detections: ${data.detections}</p>`;
                    if(data.image_path) {
                        html += `<img src="${data.image_path}" alt="Result">`;
                    }
                } else {
                    html += `<p style="color: red;">Error: ${data.error}</p>`;
                }
                html += '</div>';
                document.getElementById('results').innerHTML = html;
            };
        </script>
    </body>
    </html>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and perform YOLO detection"""
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read image
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        
        # Run YOLO inference
        results = model(img_array)
        detections = len(results[0].boxes) if results else 0
        
        # Get annotated frame from YOLO
        annotated_frame = results[0].plot() if results else img_array.copy()
        
        # Save the result image
        result_img = Image.fromarray(annotated_frame)
        output_path = os.path.join('runs', f'detection_{id(file)}.jpg')
        result_img.save(output_path)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image_path': f'/results/{os.path.basename(output_path)}',
            'detection_results': str(results[0].summary()) if results else 'No detections'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/results/<filename>')
def get_result(filename):
    """Serve result images"""
    return send_file(os.path.join('runs', filename), mimetype='image/jpeg')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
