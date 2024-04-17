import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

# using prebuilt in classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = True
detection_stopped_time = None
timer_started = False

frame_size = (int(cap.get(3)), int(cap.get(4)))  # Corrected syntax
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Corrected function name
out = cv2.VideoWriter("video.mp4", fourcc, 20, frame_size)

while True:
    _, frame = cap.read()

    # Classifiers require a greyscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Only start recording once a body or face detected
    if len(faces) + len(bodies) > 0:
        detection = True

    out.write(frame)

    # Drawing rectangles on the frame
    # for(x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y + height), (255, 0, 0), 3) #BGR

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destoryAllWindows()
