import math
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

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
face_loc=[]
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
if image is None:
	print("DOSYA UYGUN DEĞİL")
	exit()

image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)


sonuc_arr = []
sad_distances = []
happy_distances = []
def euclidian_square(a,b):
    for i in range(0,len(b)):
        param = b[i]-a[i]
        sq_par = param*param
        sonuc_arr.append(sq_par)
    sump = sum(sonuc_arr)

    del sonuc_arr[:]
    return sump

def nearest_index(uzakliklar):
    sonuc_indeks = 0
    en_kucuk_sayi = uzakliklar[0]
    for x in range(0,len(uzakliklar)):
        if en_kucuk_sayi >= uzakliklar[x]:
            en_kucuk_sayi = uzakliklar[x]

    for i in range(0,len(uzakliklar)):
        if uzakliklar[i] == en_kucuk_sayi:
            sonuc_indeks = i

    return sonuc_indeks


def ogrenim_kumesine_uzakliklari_hesapla(test_oznitelik_vektoru, egitim_oznitelik_vektorleri):
    uzakliklar = []

    for i in range(0,len(ornek_egitim_oznitelik_vektorleri)):
        oklid_kare(test_oznitelik_vektoru, egitim_oznitelik_vektorleri[i])
        uzakliklar.append(oklid_kare(test_oznitelik_vektoru, egitim_oznitelik_vektorleri[i]))
    return uzakliklar








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
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		face_loc.append(x)
		face_loc.append(y)

with open("sad.md", "rb") as fp:
    sad_val = pickle.load(fp)
with open("happy.md", "rb") as fp:
    happy_val = pickle.load(fp)


i = 0



for face in happy_val:
    happy_distances.append(euclidian_square(face_loc, face))


i=0

for face in sad_val:
    sad_distances.append(euclidian_square(face_loc, face))




srt_sad = sad_distances[:]
srt_sad.sort()
srt_happy = happy_distances[:]
srt_happy.sort()
sad_ftw = 0
happy_ftw = 0
knn = 3
combine = sad_distances[:] + happy_distances[:]
combine.sort()

for knn in range(0,3):
    for i in range(0,len(sad_distances)):
        if combine[knn] == sad_distances[i]:
            sad_ftw += 1
            break
        elif combine[knn] == happy_distances[i]:
            happy_ftw += 1
            break



if happy_ftw > sad_ftw:
    print("happy")
elif happy_ftw < sad_ftw:
    print("sad")
else:
    print("oh shit")
