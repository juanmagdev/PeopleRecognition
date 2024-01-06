from flask import Flask, render_template, Response
import cv2


app = Flask(__name__)
cam = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('camera.html')

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True)

if __name__ == "__main__":
    app.run(debug=False)