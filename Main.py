import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

# using prebuilt in classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)

    # Classifiers require a greyscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destoryAllWindows()

