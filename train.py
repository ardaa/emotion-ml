
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import tensorflow as tf
import math
import os
import pickle

face_loc = []
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
arr_sad = []
arr_happy = []
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
path_sad = "./data/sad/"


for img in os.listdir(path_sad):
    temp_img_path = path_sad+img
    print(temp_img_path)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(temp_img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)

    	# convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    	# show the face number
    	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
    	for (x, y) in shape:
    		face_loc.append(x)
    		face_loc.append(y)
    if len(face_loc) == 136:
        arr_sad.append(face_loc)
    face_loc = []

with open("sad.md", "wb") as fp:
    pickle.dump(arr_sad, fp)


path_happy = "./data/happy/"
face_loc = []

for img in os.listdir(path_happy):
    temp_img_path = path_happy+img
    print(temp_img_path)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(temp_img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)

    	# convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    	# show the face number
    	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
    	for (x, y) in shape:
    		face_loc.append(x)
    		face_loc.append(y)
    if len(face_loc) == 136:
        arr_happy.append(face_loc)
    face_loc = []
with open("happy.md", "wb") as fp:   #Pickling
    pickle.dump(arr_happy, fp)

    # show the output image with the face detections + facial landmarks
