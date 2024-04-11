import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import cmath


# Initialize the Video
cap = cv2.VideoCapture('E:/Videos/vid(2).mp4')
hsvVals = {'hmin': 3, 'smin': 120, 'vmin': 106, 'hmax': 7, 'smax': 163, 'vmax': 255}

# Create the color finder object
myColorFinder = ColorFinder(False)

# variables
posList = []
posListX, posListY = [], []
xList = [item for item in range(1000, 1300)]
prediction = False

while True:
    # Grab the image

    success, img = cap.read()
    # img = cv2.imread("E:/Ball3.png")
    # img = img[0:900, :]

    # find the color of ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # find Location of ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    # Now we are going to track the ball for current frame
    # if contours:
    #     cx, cy = contours[0]['center']
    #
    #     imgCurrent = cv2.circle(imgContours, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    #
    # # Now we are going to track the ball for every frame
    # if contours:
    #     posList.append(contours[0]['center'])
    #
    #     for pos in posList:
    #         cv2.circle(imgContours, pos, 5, (0, 255, 0), cv2.FILLED)
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    # polynomial best fit line
    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        # Find the Coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)

            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)

            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)
        #     pink line

        # Prediction
        if len(posListX) < 10:
            #   X values 950 1025   241
            a = A
            b = B
            c = C - 590

            x = int((-b + math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))

            prediction = 950 < x < 1025
        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 200, 0), offset=20)

    # Display for  ball tracking
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow('ImageContours', imgContours)
    cv2.waitKey(100)

    # # Display for ball detecting
    # imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    # cv2.imshow('ImageColor', imgColor)
    # cv2.waitKey(50)
