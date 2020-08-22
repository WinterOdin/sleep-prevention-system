import cv2
import numpy
import dlib as dl
from math import hypot
import time
from playsound import playsound


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

def blinking(pointsList, points):
        #left eye
        leftPointLine    = (points.part(pointsList[0]).x, points.part(pointsList[0]).y)
        rightPointLine   = (points.part(pointsList[3]).x, points.part(pointsList[3]).y)
        center_top       = middlepoint(points.part(pointsList[1]),points.part(pointsList[1]))
        center_bottom    = middlepoint(points.part(pointsList[5]), points.part(pointsList[5]))


        # X lines
        #we are using x lines just for visual purpose because the look cooler than horizontal and vertical
        topPointLeft     = (points.part(pointsList[2]).x, points.part(pointsList[2]).y)
        bottomPointRight = (points.part(pointsList[5]).x, points.part(pointsList[5]).y)
        topPointRight    = (points.part(pointsList[1]).x, points.part(pointsList[1]).y)
        bottomPointLeft  = (points.part(pointsList[4]).x, points.part(pointsList[4]).y)

        horLineHeight    = hypot((leftPointLine[0]-rightPointLine[0]),(leftPointLine[1]-rightPointLine[1]))
        verLineHeight    = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))

        lineX1           = cv2.line(camWindow, topPointLeft, bottomPointRight, (0,225,0),2)
        lineX2           = cv2.line(camWindow, topPointRight, bottomPointLeft, (0,225,0),2)
        
        ratio            = horLineHeight/verLineHeight
        return ratio 

#displaying camera input
while True:
    _, camWindow = camera.read()
    #for saving some computation
    grayFilter   = cv2.cvtColor(camWindow, cv2.COLOR_BGR2GRAY)
    facePoints   = faceDetector(grayFilter)
    for x in facePoints:
        pointsPredictor     = predictor(grayFilter,x)
        ratioFunctionLeft   = blinking([36,37,38,39,40,41], pointsPredictor)
        ratioFunctionRight  = blinking([42,43,44,45,46,47], pointsPredictor)
        blinkingRationAVG   = (ratioFunctionLeft+ratioFunctionRight)/2
        start = time.time()
        if blinkingRationAVG > 5.7:
            cv2.putText(camWindow,"CLOSED",(50,150),font,3,(255,0,0))
        end = time.time()
        totalBlinked = end-start
        if totalBlinked > 0.6000000000000000000:
            playsound('audio.mp3')

                    
        

    cv2.imshow('camera output',camWindow)

    key      = cv2.waitKey(1)
    if key   == 27: #27 is the ESC key on keyboard
        break
camera.release()
cv2.destroyAllWindows()
