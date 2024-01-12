import cv2
import numpy as np
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time


# Function to send email with attachment
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

# Import the necessary libraries


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

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open video file or capture device.
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# Email configuration
sender_email = "alarma970@gmail.com"
receiver_email = "manuelrr91@gmail.com"
email_password = "Cuidao!:0"

# Flag and time tracking variables
email_sent = False
last_email_time = time.time()  # Initialize with the current time

# Define the maximum time allowed between emails (in seconds)
max_time_without_detection = 300  # 5 minutes

while True:
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    person_detected = False  # Track if a person is detected in the current frame

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args.thr and confidence >=0.95:
            class_id = int(detections[0, 0, i, 1])

            if class_id == 15:  # Class 15 corresponds to 'person'
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

                label = "Person: " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label)

                person_detected = True  # Set the flag if a person is detected

    # Check if an email has not been sent and a person is detected
    if not email_sent and person_detected:
        # Send an email with the captured frame
        send_email(sender_email, email_password, receiver_email, "Person Detected", "Person detected!", frame)
        email_sent = True  # Set the flag to True after sending the email
        last_email_time = time.time()  # Update the last sent time

    # Check if the maximum time without detection has passed
    if email_sent and (time.time() - last_email_time > max_time_without_detection):
        email_sent = False  # Reset the flag after the specified duration

    # Display the frame
    cv2.imshow("frame", frame)

    # Check for the 'ESC' key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
