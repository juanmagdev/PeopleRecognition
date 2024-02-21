# Import necessary libraries
from flask import Flask, render_template, Response, request
import cv2
import argparse
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import pygame

# Define global variables
scaleFactor = 0.5
email_sent = False
last_email_time = time.time()
max_time_without_detection = 5
person_detected = False
people_detection_enabled = False
email_alarm_enabled = False
alarm_played=False

def initialize_sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("/home/manu/Desktop/PeopleRecognition/StreamAppv1/burglar-alarm.mp3")  # Replace "alarm_sound.wav" with your audio file path

# Function to play the sound alarm
def play_sound_alarm():
    pygame.mixer.music.play()

# Initialize the sound alarm
initialize_sound_alarm()

# Construct the argument parse
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, the camera's stream will be used")
parser.add_argument("--prototxt", default="/home/manu/Desktop/PeopleRecognition/PersonObjectsDetection/MobilNet_SSD_opencv-master/MobileNetSSD_deploy.prototxt",
                    help='Path to text network file: '
                         'MobileNetSSD_deploy.prototxt for Caffe model or ')
parser.add_argument("--weights", default="/home/manu/Desktop/PeopleRecognition/PersonObjectsDetection/MobilNet_SSD_opencv-master/MobileNetSSD_deploy.caffemodel",
                    help='Path to weights: '
                         'MobileNetSSD_deploy.caffemodel for Caffe model or ')
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()


# Define labels of Network
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Email configuration
sender_email = "alarma970@gmail.com"
receiver_email = "alarma970@gmail.com"
email_password = "agmn igzg auqf pang "


# Initialize the camera
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
# Get video properties
fps = int(cam.get(cv2.CAP_PROP_FPS))
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

# Initialize the Flask app
app = Flask(__name__)

# Function to send email
def send_email(sender_email, password, receiver_email, subject, body, attachment):
    msg = MIMEMultipart()
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach the image as an attachment
    _, img_encoded = cv2.imencode('.jpg', attachment)
    img_data = img_encoded.tostring()
    image = MIMEImage(img_data, name="person_detected.jpg")
    msg.attach(image)

    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

# Flask route for the index page
@app.route('/')
def index():
    return render_template('camera.html')

# Flask route for the video stream
@app.route('/video')
def video():
    global person_detected, email_sent, last_email_time, people_detection_enabled, email_alarm_enabled, sound_alarm_enabled

    people_detection_enabled = request.args.get('people_detection') == 'on'
    email_alarm_enabled = request.args.get('email_alarm') == 'on'
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame',
                    content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True)

# Function to generate video frames
def generate_frames():
    global people_detection_enabled, email_alarm_enabled, person_detected, email_sent, last_email_time
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame_small = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            person_detected=False
            if people_detection_enabled:
                # Load the Caffe model 
                net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
                frame_resized = cv2.resize(frame_rgb, (300, 300))  # Resize frame for prediction
                blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
                # Set the input blob for the network
                net.setInput(blob)
                # Perform prediction
                detections = net.forward()

                cols = frame_resized.shape[1] 
                rows = frame_resized.shape[0]

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]  # Confidence of prediction 
                    if confidence > args.thr and confidence >= 0.8:  # Filter prediction 
                        class_id = int(detections[0, 0, i, 1])  # Class label

                        if class_id == 15:
                            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                            yLeftBottom = int(detections[0, 0, i, 4] * rows)
                            xRightTop = int(detections[0, 0, i, 5] * cols)
                            yRightTop = int(detections[0, 0, i, 6] * rows)
            
                            heightFactor = frame.shape[0] / 300.0  
                            widthFactor = frame.shape[1] / 300.0 
                            xLeftBottom = int(widthFactor * xLeftBottom) 
                            yLeftBottom = int(heightFactor * yLeftBottom)
                            xRightTop = int(widthFactor * xRightTop)
                            yRightTop = int(heightFactor * yRightTop)
                            
                            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                            
                            if class_id in classNames:
                                label = classNames[class_id] + ": " + str(confidence)
                                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                                yLeftBottom = max(yLeftBottom, labelSize[1])
                                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]), (xLeftBottom + labelSize[0], yLeftBottom + baseLine), (255, 255, 255), cv2.FILLED)
                                cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                                print(label) 
                                person_detected = True
                                print(person_detected)
                                print(email_alarm_enabled)
                                print(email_sent)                       
                out.write(frame)
                if person_detected and email_alarm_enabled and not email_sent:
                        play_sound_alarm()
                        print("sending email")
                        send_email(sender_email, email_password, receiver_email, "Person Detected", "Person detected!", frame)
                        email_sent = True
                        last_email_time = time.time()  # Update the last sent time
                        print("email sent")
                if email_sent and (time.time() - last_email_time > max_time_without_detection):
                    email_sent = False  # Reset the flag after the specified duration

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#cam.release()
#out.release()

if __name__ == "__main__":
    #app.run(debug=False)
    app.run(host='0.0.0.0', port=5000)
