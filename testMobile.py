import cv2
import requests
import base64

cam = cv2.VideoCapture("rtsp://admin:admin@192.168.1.119:1935")
server_url = 'http://localhost:3001/api/upload'  
while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # guardo en un archivo la imagen
    cv2.imwrite('image.jpg', frame1)
    # guardo en un txt el base64 de la imagen
    with open("image.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())


    response = requests.post(server_url, files={'image': ('image.jpg', encoded_string)})


    if response.status_code != 200:
        raise Exception('Error: {}'.format(response.status_code))

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('Granny Cam', frame1)