from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.utils.common import read_yaml
from PIL import Image
import io
import base64
import os
from pathlib import Path
import yaml





app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Model configuration
CLASS_MAPPING = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4

base_dir = os.path.dirname(os.path.abspath(__file__))

model_config = read_yaml(Path(base_dir) / 'config' / 'config.yaml')

# Load model
model = load_model(model_config['training']['best_trained_model_path'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(file_stream):
    """Process grayscale MRI image to match model input requirements"""
    try:
        # Load as grayscale and resize
        img = Image.open(io.BytesIO(file_stream)).convert('L')
        img = img.resize((140, 140))
        
        # Convert to array with proper dimensions
        img_array = image.img_to_array(img)  # (140, 140)
        img_array = np.expand_dims(img_array, axis=-1)  # (140, 140, 1)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 140, 140, 1)
        img_array /= 255.0  # Normalize
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def predict_tumor(file_stream):
    """Make prediction on preprocessed image"""
    try:
        img_array = preprocess_image(file_stream)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = round(100 * np.max(predictions[0]), 2)
        class_name = CLASS_NAMES[predicted_class_idx]
        
        # Format class name for display
        display_name = class_name.capitalize()
        if class_name == 'notumor':
            display_name = 'No Tumor'
            
        return display_name, confidence, class_name
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        try:
            file_stream = file.read()
            
            # Verify image is grayscale
            img = Image.open(io.BytesIO(file_stream))
            if img.mode != 'L':
                img = img.convert('L')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                file_stream = img_byte_arr.getvalue()
            
            # Get prediction
            display_name, confidence, class_name = predict_tumor(file_stream)
            
            # Prepare image for display (convert to RGB)
            display_img = Image.open(io.BytesIO(file_stream)).convert('RGB')
            img_byte_arr = io.BytesIO()
            display_img.save(img_byte_arr, format='JPEG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            return render_template('result.html', 
                                result=display_name,
                                confidence=confidence,
                                original_class=class_name,
                                img_data=img_base64)
            
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('home'))
    else:
        flash('Allowed file types: PNG, JPG, JPEG')
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)