from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from deepface import DeepFace
import cv2
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return 'Welcome to Emotion Detection'

def predict_emotion(image_path):
    """
    Predicts the dominant emotion from the given image.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - str: The predicted dominant emotion.
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            error_message = "Error: Unable to read the image. Please check the file format or path."
            print(error_message)  # Print error to the server console
            return error_message

        # Resize the image
        img = cv2.resize(img, (224, 224))

        # Analyze the image for emotions
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # Extract the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        # print(f"Detected Emotion: {dominant_emotion}")  # Print emotion to the server console
        return dominant_emotion
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)  # Print error to the server console
        return error_message

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image file provided in the request.")  # Print error to the server console
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        print("No file selected for upload.")  # Print error to the server console
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        #  print(f"File saved at: {filepath}")  # Print file save location to the server console

        # Predict the uploaded image
        prediction = predict_emotion(filepath)
        
        return prediction

if __name__ == '__main__':
    app.run(debug=True)
