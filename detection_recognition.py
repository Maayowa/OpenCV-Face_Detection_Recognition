import numpy as np
import cv2
import time
import os
from PIL import Image

# Creating the cornered edges lines
# top Left edge: hstart = (x + rad, y), hend = (x + rad + len, y), vstart = (x, y + rad), vend = (x, y + rad + len)
# Top right edge: hstart = (x + w -rad - len, y), hend = (x + w - rad, y), vstart = (x + w, y+rad), vend = (x + w, y + rad + len)
# bot left edge: hstart = (x + rad, y + h), hend = (x + rad + len, y + h), vstart = (x + w, y + h - rad - len), vend = (x + w, y + h - rad)
# bot right edge: hstart = (x + w - rad - len, y + h), hend = (x + w - rad, y + h), vstart = (x + w , y + w - rad - len), vend = (x + w, y + w - rad)

def detect_face(img, cascade, scale_fac = 1.1, minNeighbor = 3):
    # Remember face detection would take place using edges, hence we need to change image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Should in case we have multiple faces
    faces = face_cascade.detectMultiScale(gray, scale_fac, minNeighbors = minNeighbor)
    
    return faces


def show_faces(img, faces, label = [], text = "" ,color = (255, 0, 255), clf = None, predict = False):
    """
    For face detection, it draws a rectangle around detected face and adds text,
    if included, above it. If 'predict' parameter is true, classfier and labels would be
    added for predictions to be made.

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coord = []
    w, h = gray.shape[::-1]
    fsc = max(w,h)*0.8/1000 # To scale text to image
    thick = 1
    if fsc > 2: thick = 2
    
    for (x,y,w,h) in faces:
        rad = int(0.1 * w)
        llen = rad
        axes = (rad, rad)
        angle = 0
        
        if predict:
            idx, conf_ = clf.predict(gray[y: y+h, x: x+w])
            text = label[idx]
            text += " (" + "{:.2f}".format(conf_) + "%)" 
            
            if conf_ > 90:
                text = "Unknown"

        # Top Left
        cv2.ellipse(img, (x+rad, y+rad), axes, angle, 180, 270, color, 2) # or (255, 255, 0)
        cv2.line(img, (x + rad, y), (x + rad + llen, y),color, 2)
        cv2.line(img, (x, y + rad), (x, y + rad + llen), color, 2)

            # Top right
        cv2.ellipse(img, (x + w - rad, y + rad), axes, angle, 270, 360, color, 2)
        cv2.line(img, (x + w - rad - llen, y), (x + w - rad, y), color, 2)
        cv2.line(img, (x + w, y + rad), (x + w, y + rad + llen), color, 2)

            # Bottom Left
        cv2.ellipse(img, (x + rad, y + h - rad), axes, angle, 90, 180, color, 2)
        cv2.line(img, (x + rad, y + h), (x + rad + llen, y + h),color, 2)
        cv2.line(img, (x, y + h - rad - llen), (x, y + h - rad), color, 2)

            # Bottom right
        cv2.ellipse(img, (x + w - rad, y + h - rad), axes, angle, 0, 90, color, 2)
        cv2.line(img, (x + w - rad -llen, y + h), (x + w - rad, y + h), color, 2)
        cv2.line(img, (x + w, y + h - rad - llen), (x + w, y + h - rad), color, 2)
 
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, fsc, color, thick, cv2.LINE_AA)
        
        coord = [x, y, w, h]
            
    return coord, img


def vid_predict(path, name, clf, labels, cap_duration, cascade, sc_frac = 1.2, minNN = 8): # or size
    
    """ Recognize faces in videos. 
    path = 0 for real time prediction
    """
    
    cap = cv2.VideoCapture(path)
    start_time = time.time()

    while ( int(time.time() - start_time) < cap_duration): # or while (img_id < size):
        _, img = cap.read()
        face = detect_face(img, face_cascade,scale_fac = sc_frac, minNeighbor=minNN)
        _, img = show_faces(img, face,labels, clf =  clf, predict = True)
        
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# FACE DETECTION

img = cv2.imread('people.jpg', 1)

face_cascade = cv2.CascadeClassifier('assets\\haarcascade_frontalface_default.xml')

face = 
face = detect_face(img, face_cascade,1.1, minNeighbor=5)
_, img = show_face(img, face, "Face")
cv2.imwrite('detected_people.jpg', img)
cv2.imshow('Detected Image', img)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()

# FACE RECOGNITION

# testing face recognition on youtube video
path = r"test\video\i-bought-the-bank-ending-justice-league-4k-sdr_OXqsrGxn_lIn4.mp4"
face_cascade = cv2.CascadeClassifier('assets\\haarcascade_frontalface_default.xml')

clf = cv2.face.LBPHFaceRecognizer_create()

clf.read('assets\\classifier.yml')

vid_predict(path, 'new_video', clf, os.listdir('data'), 15, face_cascade) 