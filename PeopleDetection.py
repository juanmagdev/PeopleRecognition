import numpy as np
import cv2
import time

font=cv2.FONT_HERSHEY_SIMPLEX
fpsReport=0
timeStamp=time.time()
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)
#cap= cv2.VideoCapture('rtsp://admin:admin@192.168.18.226:1935')

# the output will be written to output.avi
'''
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
'''
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsReport=.9*fpsReport+ .1*fps
    #fpsReport=fps
    cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)
    cv2.putText(frame,str(round(fpsReport,1))+ 'fps',(5,25),font,.75,(0,255,255),2)
    
    
    # Write the output video 
    #out.write(frame.astype('uint8'))


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)