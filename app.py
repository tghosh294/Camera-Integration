from flask import Flask, render_template, Response, request, jsonify
import cv2
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

# Function to generate video frames with pose estimation
def generate_frames():
    global counter, stage, probability
    cap = cv2.VideoCapture(0)  # Access the webcam
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert the frame to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image with MediaPipe Pose
        results = pose.process(image)
        
        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and calculate angles
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for the right hip, knee, and ankle
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate the angle between the hip, knee, and ankle
            angle = calculate_angle(hip, knee, ankle)
            probability = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            
            # Update the stage and counter based on the angle
            if angle > 160:
                stage = "up"
            if angle < 70 and stage == "up":
                stage = "down"
                counter += 1
        
        # Display the stage, reps, and probability on the frame
        cv2.rectangle(image, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.putText(image, f'STAGE: {stage}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'REPS: {counter}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'PROB: {probability:.2f}', (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Encode the frame as JPEG and yield it for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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