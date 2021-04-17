# Import relevant libraries
import numpy as np
import cv2
import os
from PIL import Image

def train_classifier(train_dir, save_dir, len_img):
    """
    Trains the faces provided using cv2 face classifier
    """
    
    faces = []
    ids = []
    labels = []
    
    folder_path = [os.path.join(train_dir, actor) for actor in os.listdir(train_dir)]
    names = os.listdir(train_dir)
    print("Collecting...\n")
    # loops through folder of pictures in train folder
    for idx, folder in enumerate(folder_path):
        img = []
        idxx = []
        if len(os.listdir(folder)) == 0: continue
            
        filepath = [os.path.join(folder, roi) for roi in os.listdir(folder)]
        # Loops through individual face in folder
        for roi in filepath:
            roi = Image.open(roi).convert("L") #converts to grayscale
            np_roi = np.array(roi, 'uint8')
            img.append(np_roi)
            idxx.append(idx)
        
        labels.append(name)  
        img_size = min(len(img), len_img)
        faces += img[:img_size]
        ids += idxx[:img_size]
        print(f"{img_size} images for {names[idx]}...")
        
    np.save(save_dir + '\\faces_arr.npy', faces)
    np.save(save_dir + '\\face_id.npy', ids)
    print("Creating model...") 
    
    classifier_dir = save_dir + "\\classifier.yml"
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write(classifier_dir)
    print("\nCompleted!")
        

train_classifier('data', 'assets', 100)