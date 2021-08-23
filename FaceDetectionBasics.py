import cv2
import mediapipe as mp
import time

cap = cv2. VideoCapture()

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
