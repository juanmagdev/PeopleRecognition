from flask import Flask, render_template, Response
import face_recognition
import os
import cv2
import pickle
import time

fpsReport=0
scaleFactor=0.25

Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
cam=cv2.VideoCapture(0)
#cam= cv2.VideoCapture('rtsp://admin:admin@192.168.18.226:1935')
font=cv2.FONT_HERSHEY_SIMPLEX

timeStamp=time.time()



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('camera.html')

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
            frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
            facePositions=face_recognition.face_locations(frameRGB,model='cnn') #Default hog (simpler, for raspberry) CNN much more sophisticated
            allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
            for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
                name='Unknown Person'
                matches=face_recognition.compare_faces(Encodings,face_encoding)
                if True in matches:
                    first_match_index=matches.index(True)
                    name=Names[first_match_index]
                top=int(top/scaleFactor)
                right=int(right/scaleFactor)
                left=int(left/scaleFactor)
                bottom=int(bottom/scaleFactor)
                cv2.rectangle(frame,(left,top),(right,bottom),(0,255,255),2)
                cv2.putText(frame,name,(left,top-6),font,.75,(0,255,255),2)
            
            #dt=time.time()-timeStamp
            #timeStamp=time.time()
            #fps=1/dt
            #fpsReport=.9*fpsReport+ .1*fps
            #cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)
            #cv2.putText(frame,str(round(fpsReport,1))+ 'fps',(5,25),font,.75,(0,255,255),2)
            #print('fps:',round(fpsReport,1))
            #cv2.imshow('Picture',frame)
            #cv2.moveWindow('Picture',0,0)
            
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True)

if __name__ == "__main__":
    app.run(debug=False)