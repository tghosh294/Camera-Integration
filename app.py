from flask import Flask, render_template, Response, request, jsonify
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for counting reps and tracking stage
counter = 0
stage = None
probability = 0.0

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get the current counter, stage, and probability
@app.route('/data')
def get_data():
    global stage, counter, probability
    return jsonify({
        'stage': stage,
        'reps': counter,
        'prob': probability
    })

# Route to reset the counter
@app.route('/reset', methods=['POST'])
def reset_counter():
    global counter
    counter = 0
    return "OK"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)