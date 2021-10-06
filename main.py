# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:52:39 2021

@author: 91809
"""
import cv2
import face_recognition
import numpy as np
import os
import datetime as datetime


image_path = 'training_images'
class_names = []
original_images = []
original_encodings = []

''' 
STEP-1

1. Load the training images
2. Find the encoding of each image and store in an array. This encoding would be used to compare with the input images
'''
imag_files = os.listdir(image_path)

# Load training images
for file_name in imag_files:
    original_images.append(cv2.imread(f'{image_path}/{file_name}'))
    # Get the file name (class name)
    class_names.append(os.path.splitext(file_name)[0]) # splitext(x) will split the filename.jpg to 'filename' & '.jpg'
    
# Find encoding of images    
def find_encodings(images):   
    original_encodings = []
    
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_encodings.append(face_recognition.face_encodings(image)[0])    
    return original_encodings
    
'''
STEP-2 

Define the methods to identify the face using face detection mechanism

'''
def recognize(frame, orig_encodings, class_names):
    # Resize the image
    img = cv2.resize(frame, (0,0), None, 0.5, 0.5) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find the location of faces in the given frame
    face_locs_in_current_frame = face_recognition.face_locations(img)
    print(face_locs_in_current_frame)
    
    # Find the encoding of the faces
    face_encodings_in_current_frame = face_recognition.face_encodings(img, face_locs_in_current_frame)
    
    for facelocation, faceencoding in zip(face_locs_in_current_frame, face_encodings_in_current_frame):
        # Match the face
        matches = face_recognition.compare_faces(orig_encodings, faceencoding)
        # Output for two training images scenario : [True, False]
        
        # Find the distance between faces. 
        # This info is helpful to select the best match in case there are multiple matches for an image.
        # This is also helpful to identify how similar two images are, even if they don't match.
        
        # Lower the value, better the match.
        faceDists = face_recognition.face_distance(orig_encodings, faceencoding)
        # Output for two training images scenario : [0.5666, 0.8787]
        
        # Select the index of the image which is mostly matching with the input image
        match_index = np.argmin(faceDists)
        # Ouput: 0
        
        # Check if the selected image is marked as True in the matching data
        if matches[match_index]: # output: matches[0] = True
            classname = class_names[match_index]
            print(f'Match found as "{classname}"')
            
            #
            # Show the recognized faces in the image
            #
            
            # Get the face coordinates in the resized image
            y1, x2, y2, x1 = facelocation
            # Find the face coordinates in the actual image (as we resized to 50%, now, we need to multiply 2 to each coordinate)
            y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
            
            # Draw a rectange around the face
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(frame, (x1,y2-40), (x2,y2), (0,255,0),cv2.FILLED)
            cv2.putText(frame,classname,(x1+6, y2-15), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        else:
            print('----- No face match ------')
        
        
        cv2.imshow('Webcam',frame)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

    
    
    



'''
STEP-3: 

Capture image using camera

'''
cap = cv2.VideoCapture(0)

orig_encodings = find_encodings(original_images)

while True:
    _ , frame = cap.read()
    recognize(frame, orig_encodings, class_names)    
    




