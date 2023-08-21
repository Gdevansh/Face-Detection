import cv2
import face_recognition

imgs = face_recognition.load_image_file('Image/devansh.jpg')
imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
#imgst = face_recognition.load_image_file('Image/Devansh Gupta.jpg')
#imgst = cv2.cvtColor(imgst,cv2.COLOR_BGR2RGB)
imgme = cv2.resize(imgs,(0,0),None,0.25,0.25,interpolation=cv2.INTER_AREA)
#imgTest = cv2.resize(imgst,(0,0),None,0.75,0.75,interpolation=cv2.INTER_AREA)


#print(imgs.shape)
#print(imgme.shape)

faceLoc = face_recognition.face_locations(imgme)[0]
encodeme = face_recognition.face_encodings(imgme)[0]
cv2.rectangle(imgme,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


cap = cv2.VideoCapture(0)
#cv2.imshow('Devansh Gupta', imgme)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25,)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    print(facesCurFrame)

    for encodeFace, faceLocCam in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces([encodeme], encodeFace)
        faceDis = face_recognition.face_distance([encodeme], encodeFace)
        print(matches)
        print(faceDis)

        y1,x2,y2,x1 = faceLocCam
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #cv2.putText(img, f'{matches} {round(faceDis[0], 2)} ', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
        if matches == [True]:
         cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
         cv2.putText(img,'Devansh',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodemeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTeste,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)


#results = #face_recognition.compare_faces(&#91;encodeElon], encodeTest)
#faceDis = #face_recognition.face_distance(&#91;encodeElon], encodeTest)
#cv2.putText(imgTest,f'{results} {round(faceDis&#91;0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

