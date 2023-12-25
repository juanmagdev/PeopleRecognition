import cv2
video = cv2.VideoCapture("rtsp://admin:admin@192.168.1.119:1935") # 0: webcam


while True:
    ret, frame = video.read()
    cv2.imshow('RTSP', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()