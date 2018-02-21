import numpy as np
import os
import cv2
from PIL import Image
import pickle, sqlite3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.load("trainer/training_data.yml")


def getProfile(id):
    conn=sqlite3.connect("CollgeAttendence.db")
    query="SELECT * FROM Attendence WHERE Rollno="+str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2 .circle(img, (x+w/2,y+w/2), 90, (255, 0, 255), 2)
        #cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
        # Hiding the eye detector for now
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 40:
           profile=getProfile(id)
           if(profile != None):
                cv2.putText(img, "Rollno: "+str(profile[0]), (x, y+h+30), font, .4, (0, 125, 27), 1);
                cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 60), font,  0.8, (0, 0, 255), 1);
                cv2.putText(img, "Level: " + str(profile[2]), (x, y + h + 90), font, 0.4, (0, 125, 27), 1);
                cv2.putText(img, "Faculty: " + str(profile[3]), (x, y + h + 120), font,  0.4, (0, 125, 27), 1);
                cv2.putText(img, "Semester: "+str(profile[4]), (x, y+h+150), font, 0.4, (0, 125, 27), 1);
                cv2.putText(img, "Address: " + str(profile[5]), (x, y + h + 180), font,  0.4, (0, 125, 27), 1);
                cv2.putText(img, "Phone: "+str(profile[6]), (x, y+h+210), font, 0.4, (0, 125, 27), 1);
                conn=sqlite3.connect("rai.db")
                roCursor=conn.cursor()
                roCursor=conn.execute("UPDATE rabin SET date='p' WHERE Id="+str(id));
    
                conn.commit()
                conn.close()
                ##conn=sqlite3.connect("Attendence1.db")
                ##roCursor=conn.cursor()
                ##roCursor=conn.execute("UPDATE table SET Attendence='p' WHERE Id="+str(id));
    
                ##conn.commit()
                ##conn.close()
        else:
            cv2.putText(img, "Rollno: Unknown", (x, y + h + 30), font, .4, (0, 191, 255), 1);
            cv2.putText(img, "Name: Unknown", (x, y + h + 50), font, 0.4, (0, 191, 255), 1);
            cv2.putText(img, "Level: Unknown", (x, y + h + 70), font, 0.4, (0, 191, 255), 1);
            cv2.putText(img, "Faculty: Unknown", (x, y + h + 90), font, 0.4, (0, 191, 255), 1);
            cv2.putText(img, "Semester: Unknown", (x, y + h + 110), font, 0.4, (0, 191, 255), 1);
            cv2.putText(img, "Address:Unknown: ",  (x, y + h + 130), font,  0.4, (0, 191, 255), 1);
            cv2.putText(img, "Phone:Unknown", (x, y+h+150), font, 0.4, (0, 191, 255), 1);

    cv2.imshow('img', img)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
