import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
# CORS allows your frontend (website) to talk to this backend
CORS(app)

# 1. LOAD YOUR TRAINED MODEL
# Make sure 'brain_tumor_model.h5' is in the same folder as this script!
MODEL_PATH = 'brain_tumor_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image):
    """
    This function processes the image exactly how the model expects it.
    """
    # Resize to match the (150, 150) used during training
    image = image.resize((150, 150))
    # Convert image to a numpy array
    img_array = np.array(image)
    # Rescale pixels (0-255 to 0-1)
    img_array = img_array / 255.0
    # Add an extra dimension because the model expects a "batch" of images
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was actually sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Open the image file
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess the image
        processed_img = prepare_image(img)
        
        # MAKE THE PREDICTION
        prediction = model.predict(processed_img)
        
        # The output of a 'sigmoid' layer is a probability between 0 and 1
        # Usually: > 0.5 means "Yes (Tumor)", < 0.5 means "No (Healthy)"
        probability = float(prediction[0][0])
        result = "Tumor Detected" if probability > 0.5 else "No Tumor"
        
        # Send the answer back to your website
        return jsonify({
            'prediction': result,
            'confidence': round(probability if probability > 0.5 else (1 - probability), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on your local computer
    print("Backend server starting...")
    app.run(debug=True, port=5000) 