# Import relevant libraries
import cv2
import time
import os

def detect_face(img, cascade, scale_fac = 1.1, minNeighbor = 3):
    # Remember face detection would take place using edges, hence we need to change image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Should in case we have multiple faces
    faces = face_cascade.detectMultiScale(gray, scale_fac, minNeighbors = minNeighbor)
    
    return faces


def generate_dataset(path, name, cap_duration): # or size
    
    """Collect images for train data from videos with the aim of getting different angles 
    of the target owing to face movement during speech
    """
    
    cap = cv2.VideoCapture(path)
    dirname = f'data\\{name}'
    os.mkdir(dirname)

    img_id = 0
    start_time = time.time()

    while ( int(time.time() - start_time) < cap_duration): # or while (img_id < size):
        _, img = cap.read()
        face = detect_face(img, face_cascade,scale_fac = 1.3, minNeighbor=5)
        for coord in face:
            if len(coord) == 4:
                filename = f"{name}_{img_id}.jpg"
                roi = img[coord[1]:coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
                cv2.imwrite(os.path.join(dirname, filename), roi)
            
        img_id += 1

    cap.release()
    cv2.destroyAllWindows()
    

face_cascade = cv2.CascadeClassifier('assets\\haarcascade_frontalface_default.xml')

for video in os.listdir('videos'):
    name = '_'.join((video.split('\\')[-1]).split('-')[:2])
    path = "videos\\" + video 
    generate_dataset(path, name, 6)
    
