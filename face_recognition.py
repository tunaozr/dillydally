import numpy as np
import cv2 as cv
import os

people = []

for i in os.listdir(r'/Users/tunaozr/Desktop/facerec/train'):
    if not i.startswith('.'):
        people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'/Users/tunaozr/Desktop/valid/lebron-james-photo-by-streeter-lecka_getty-images.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (90,90), cv.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)


cv.imshow('detected face', img)
cv.waitKey(0)
