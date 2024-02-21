import numpy as np
import argparse
import cv2
import base64
import smtplib
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Function to send email with attachment
def send_email(sender_email, password, receiver_email, subject, body, attachment):
    msg = MIMEMultipart()
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach the image as an attachment
    _, img_encoded = cv2.imencode('.jpg', attachment)
    img_data = img_encoded.tobytes()
    image = MIMEImage(img_data, name="person_detected.jpg")
    msg.attach(image)

    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())