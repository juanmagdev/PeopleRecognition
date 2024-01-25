from flask import Flask, render_template, Response, request, redirect, url_for
import face_recognition
import os
import argparse
import cv2
import pickle
import time

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="/home/manu/Desktop/PeopleRecognition/PersonObjectsDetection/MobilNet_SSD_opencv-master/MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="/home/manu/Desktop/PeopleRecognition/PersonObjectsDetection/MobilNet_SSD_opencv-master/MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

scaleFactor = 0.25

Encodings = []
Names = []

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
timeStamp = time.time()

app = Flask(__name__)

facial_detection_enabled = False  # Default is disabled
people_detection_enabled = False

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video')
def video():
    global facial_detection_enabled
    global people_detection_enabled

    facial_detection_enabled = request.args.get('facial_detection') == 'on'
    people_detection_enabled = request.args.get('people_detection') == 'on'

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True)

def generate_frames():
    global facial_detection_enabled
    global people_detection_enabled
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frameSmall = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
            frameRGB = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)

            if facial_detection_enabled:
                facePositions = face_recognition.face_locations(frameRGB, model='cnn')
                allEncodings = face_recognition.face_encodings(frameRGB, facePositions)

                for (top, right, bottom, left), face_encoding in zip(facePositions, allEncodings):
                    name = 'Unknown Person'
                    matches = face_recognition.compare_faces(Encodings, face_encoding)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = Names[first_match_index]

                    top = int(top / scaleFactor)
                    right = int(right / scaleFactor)
                    left = int(left / scaleFactor)
                    bottom = int(bottom / scaleFactor)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, name, (left, top - 6), font, .75, (0, 255, 255), 2)
            if people_detection_enabled:
                    #Load the Caffe model 
                    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
                    frame_resized = cv2.resize(frameRGB,(300,300)) # resize frame for prediction
                    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
                    #Set to network the input blob 
                    net.setInput(blob)
                    #Prediction of network
                    detections = net.forward()

                    #Size of frame resize (300x300)
                    cols = frame_resized.shape[1] 
                    rows = frame_resized.shape[0]

                    #For get the class and location of object detected, 
                    # There is a fix index for class, location and confidence
                    # value in @detections array .
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2] #Confidence of prediction 
                        if confidence > args.thr: # Filter prediction 
                            class_id = int(detections[0, 0, i, 1]) # Class label

                            if class_id==15:

                                # Object location 
                                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                                xRightTop   = int(detections[0, 0, i, 5] * cols)
                                yRightTop   = int(detections[0, 0, i, 6] * rows)
            
                                # Factor for scale to original size of frame
                                heightFactor = frame.shape[0]/300.0  
                                widthFactor = frame.shape[1]/300.0 
                                # Scale object detection to frame
                                xLeftBottom = int(widthFactor * xLeftBottom) 
                                yLeftBottom = int(heightFactor * yLeftBottom)
                                xRightTop   = int(widthFactor * xRightTop)
                                yRightTop   = int(heightFactor * yRightTop)
                                # Draw location of object  
                                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                            (0, 255, 0))

                                # Draw label and confidence of prediction in frame resized
                                if class_id in classNames:
                                    label = classNames[class_id] + ": " + str(confidence)
                                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                                    yLeftBottom = max(yLeftBottom, labelSize[1])
                                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                                        (255, 255, 255), cv2.FILLED)
                                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                                    print(label) #print class and confidence

                    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                    #cv2.imshow("frame", frame)



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/add_person', methods=['POST'])
def add_person():
    try:
        # Obtain data from the form
        person_name = request.form['personName']
        uploaded_file = request.files['imageFile']

        # Print debug information
        print(f"Received Form Data - Person Name: {person_name}")

        if person_name and uploaded_file:
            # Save the uploaded image to a temporary file
            temp_image_path = f"temp_images/{person_name}.jpg"
            print(f"Temp Image Path: {temp_image_path}")
            if not os.path.exists('temp_images'):
                os.makedirs('temp_images')


            # Save the file using the save() method
            uploaded_file.save(temp_image_path)

            # Load the uploaded image and perform face encoding
            if not os.path.exists('temp_images'):
                os.makedirs('temp_images')

            person = face_recognition.load_image_file(temp_image_path)
            encoding = face_recognition.face_encodings(person)[0]

            # Update the existing Encodings and Names lists
            Encodings.append(encoding)
            Names.append(person_name)

            # Save the updated lists to the train.pkl file
            with open('train.pkl', 'wb') as f:
                pickle.dump(Names, f)
                pickle.dump(Encodings, f)

            print("Person successfully added!")
            return redirect(url_for('index'))

        else:
            print("Person name or image file missing in the form.")
            return "Failed to add person. Please provide a name and an image file."

    except Exception as e:
        print(f"Error adding person: {e}")
        return "Failed to add person. An error occurred."

if __name__ == "__main__":
    app.run(debug=False)
