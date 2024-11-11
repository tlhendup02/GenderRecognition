# app.py
import os
from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler - you'll need to save this during training
# Add this after your training code:
# joblib.dump(scaler, 'scaler.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        [np.mean(spectral_centroid)],
        [np.mean(spectral_bandwidth)],
        [np.mean(spectral_rolloff)],
        [np.mean(zero_crossing_rate)]
    ])
    
    return features.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.wav'):
            return jsonify({'error': 'Please upload a WAV file'}), 400
        
        # Extract features
        features = extract_features(file)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        gender = "Female" if prediction[0][0] < 0.5 else "Male"
        confidence = float(prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0])
        
        return jsonify({
            'prediction': gender,
            'confidence': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)