# FACIAL EMOTION DETECTION WITH LANDMARKS AND KNN

This project was from Summer 2018 but I always wanted to improve it. Ain't nobody got time for that so... 

You will need 2 sets of images, happy and sad. (If you want to retrain)

`path_happy = "./data/happy/"
path_sad = "./data/sad/"`

And you will need a dataset of the library that I am using (can't remember). 
https://drive.google.com/file/d/1JocJFmta2eJKVQRC5eRlCX0mzZoUdRVI/view?usp=sharing

*WARNING*
Includes sloppy code, some turkish code that i copypasta'd from another project of mine and was too lazyass to translate, and a free Coca-Cola(no).
It works quite well actually

Train:
`pythonw train.py -p shape_predictor_68_face_landmarks.dat`
Predict:
`pythonw predict.py -p shape_predictor_68_face_landmarks.dat -i ./i_am_a_fucking_image.jpg`
