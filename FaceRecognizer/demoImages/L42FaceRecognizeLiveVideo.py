import face_recognition
import os
import cv2
import pickle
print(cv2.__version__)

Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
cam=cv2.VideoCapture(0)
#cam= cv2.VideoCapture('rtsp://admin:admin@192.168.18.211:1935')
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=.33,fy=.33)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB,model='cnn') #Default hog (simpler, for raspberry) CNN much more sophisticated
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name='Unknown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
        top=top*3
        right=right*3
        left=left*3
        bottom=bottom*3
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,255,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows

