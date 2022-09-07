from ast import match_case
from colorsys import rgb_to_hls
from platform import release
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "images"
images = []
personName = []

myList = os.listdir(path)
print(myList)
for cu_img in myList:
    curr_img=cv2.imread(f'{path}/{cu_img}')
    images.append(curr_img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)

def faceEncodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

getencodelist=(faceEncodings(images))
print("ALL ENCODING COMPLETE")

def attendance(name):
    with open('attendance.csv','r+') as f:
        myDatalist = f.readlines()
        namelist = []
        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d:%m:%Y')
            f.writelines(f'{name},{tStr},{dStr}')



cam = cv2.VideoCapture(0)
while True :
    ret, frame = cam.read()
    faces = cv2.resize(frame,(0,0),None,0.25,0.25)
    faces = cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

    faces_curr_frame= face_recognition.face_locations(faces)
    encode_curr_frame = face_recognition.face_encodings(faces,faces_curr_frame)

    for encode_face , face_loc in zip(encode_curr_frame,faces_curr_frame):
        matches = face_recognition.compare_faces(getencodelist,encode_face)
        faceDis = face_recognition.face_distance(getencodelist,encode_face)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = face_loc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,100,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            attendance(name)

    cv2.imshow("Camera",frame)
    if cv2.waitKey(1) == 13 :
        break
cam.release()
cv2.destroyAllWindows()
