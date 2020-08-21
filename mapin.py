import cv2
import numpy
import dlib as dl
from math import hypot




# 0 stands for first camera that you have plugged in 
# if you have more than one input source you can switch beetwen them by changing the value in VideoCapture brackets
camera = cv2.VideoCapture(0)

#face detector 
faceDetector     = dl.get_frontal_face_detector()
predictor        = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

#we need this for the vertical line in our eye
def middlepoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
font = cv2.FONT_HERSHEY_COMPLEX
#displaying camera input
while True:
    _, camWindow = camera.read()
    #for saving some computation
    grayFilter   = cv2.cvtColor(camWindow, cv2.COLOR_BGR2GRAY)

    facePoints   = faceDetector(grayFilter)
    for x in facePoints:
        i, j   = x.left(),x.top()
        i1, j1 = x.right(),x.bottom()
        cv2.rectangle(camWindow,(i,j),(i1,j1), (0,0,255),3)
        points = predictor(grayFilter,x)


        #left eye
        leftPointLine    = (points.part(36).x, points.part(36).y)
        rightPointLine   = (points.part(39).x, points.part(39).y)
        center_top       = middlepoint(points.part(37),points.part(38))
        center_bottom    = middlepoint(points.part(41), points.part(40))


        # X lines
        #we are using x lines just for visual purpose because the look cooler than horizontal and vertical
        topPointLeft     = (points.part(38).x, points.part(38).y)
        bottomPointRight = (points.part(41).x, points.part(41).y)
        topPointRight    = (points.part(37).x, points.part(37).y)
        bottomPointLeft  = (points.part(40).x, points.part(40).y)
       
        horLineHeight    = hypot((leftPointLine[0]-rightPointLine[0]),(leftPointLine[1]-rightPointLine[1]))
        verLineHeight    = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
        
        ratio            = horLineHeight/verLineHeight
        #customize this to suite your eye
        if ratio >4.3:
            cv2.putText(camWindow,'BLINKING',(50,150),font,3,(255,0,0))

        lineX1  = cv2.line(camWindow, topPointLeft, bottomPointRight, (0,225,0),2)
        lineX2  = cv2.line(camWindow, topPointRight, bottomPointLeft, (0,225,0),2)

    cv2.imshow('camera output',camWindow)

    key      = cv2.waitKey(1)
    if key   == 27: #27 is the ESC key on keyboard
        break
camera.release()
cv2.destroyAllWindows()
