import cv2
import mediapipe as mp
import time
import math
import numpy as np


def l2_dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


class handDetector():
    def __init__(self, mode=False, maxHands=1, detectConf=0.65, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingerTops = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True, handNo=0):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.multi_hand_landmarks[0], self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        xCoorList = []
        yCoorList = []
        bbox = []
        self.landmarksList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                # print(id, landmark)
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                xCoorList.append(cx)
                yCoorList.append(cy)
                # print(id, cx, cy)
                self.landmarksList.append([id, cx, cy])

        xmin, xmax = min(xCoorList), max(xCoorList)
        ymin, ymax = min(yCoorList), max(yCoorList)
        bbox = xmin, ymin, xmax, ymax

        return self.landmarksList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb, we will compare x, not y
        if self.landmarksList[self.fingerTops[0]][1] > self.landmarksList[self.fingerTops[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers, y will be taken
        for id in range(1, 5):
            if self.landmarksList[self.fingerTops[id]][2] < self.landmarksList[self.fingerTops[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def checkErase(self):
        count = 0
        for id in range(1, 5):
            if self.landmarksList[self.fingerTops[id]][2] > self.landmarksList[0][2]:
                count += 1
        if count == 4:
            return True
        else:
            return False

    def checkDraw(self):
        dist = []
        for id in range(0, 5):
            dist.append(l2_dist(self.landmarksList[self.fingerTops[id]][1],
                        self.landmarksList[self.fingerTops[id]][2],  self.landmarksList[0][1], self.landmarksList[0][2]))
        if np.max(dist) == dist[1]:
            return True
        else:
            return False
