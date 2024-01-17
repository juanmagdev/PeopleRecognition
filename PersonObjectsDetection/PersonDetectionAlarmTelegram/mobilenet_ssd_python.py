import numpy as np
import argparse
import cv2
from telegram import Bot, InputFile
import time
import asyncio
import os
import base64
from io import BytesIO
from PIL import Image

telegram_bot_token = ''
bot = Bot(token=telegram_bot_token)
chat_id = '1027514141'
message_text = 'Cuidao que te roban'

async def send_telegram_message(text, image_base64):
    try:
        # Decode base64 image data
        image_binary = base64.b64decode(image_base64)

        # Convert binary image data to a BytesIO stream
        image_stream = BytesIO(image_binary)

        # Open the image using PIL (Python Imaging Library)
        image = Image.open(image_stream)

        # Create a new BytesIO stream to save the image as JPEG
        jpeg_stream = BytesIO()
        image.save(jpeg_stream, format='JPEG')

        # Send the JPEG image as a document with the specified caption
        await bot.send_document(chat_id=chat_id, document=InputFile(jpeg_stream.getvalue(), filename='image.jpg'), caption=text)

    except Exception as e:
        print(f"Error sending message: {e}")
        
async def main():
    # Open video file or capture device.
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

    # Flag and time tracking variables
    message_sent = False
    last_message_time = time.time()  # Initialize with the current time

    # Define the maximum time allowed between emails (in seconds)
    max_time_without_detection = 5  # time in seconds

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

            if confidence > args.thr and confidence >= 0.8:
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

        # Check if a message has not been sent and a person is detected
        if not message_sent and person_detected:
            # Encode the image as base64
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(img_encoded).decode('utf-8')



            # Send the image as a document
            await send_telegram_message(message_text, image_base64)
            message_sent = True  # Set the flag to True after sending the email
            last_email_time = time.time()  # Update the last sent time

        # Check if the maximum time without detection has passed
        if message_sent and (time.time() - last_email_time > max_time_without_detection):
            message_sent = False  # Reset the flag after the specified duration

        # Display the frame
        cv2.imshow("frame", frame)

        # Check for the 'ESC' key to exit the loop
        if cv2.waitKey(1) >= 0:
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    asyncio.run(main())
