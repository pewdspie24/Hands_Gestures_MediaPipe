import os
import math
from datetime import datetime
import argparse
import cv2
import numpy as np
import handModule as htm

def main(reso):
    #######################
    brushThickness = 1
    drawColor = (255, 0, 255)
    headerColor = [(255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 0, 0)]
    ########################

    # HD webcam will be (720, 1280, 3), sorry but my webcam support only for standard VGA :(
    if reso == 'vga':
        WINDOW_SIZE = (480, 640, 3)
    else:
        WINDOW_SIZE = (720, 1280, 3)
    HEADER_SIZE = (int(WINDOW_SIZE[1]), int(WINDOW_SIZE[0]/20))

    imgCanvas = np.zeros(WINDOW_SIZE, np.uint8)
    # print(WINDOW_SIZE[:-1]/4)
    header = np.zeros(WINDOW_SIZE, np.uint8)
    header[:] = headerColor[1]
    # print(header.shape)
    header[0:WINDOW_SIZE[0], 0:(WINDOW_SIZE[1]//4)] = headerColor[0]
    header[0:WINDOW_SIZE[0], (WINDOW_SIZE[1]//4)
                            :(WINDOW_SIZE[1]//2)] = headerColor[1]
    header[0:WINDOW_SIZE[0], (WINDOW_SIZE[1]//2)
                            :(WINDOW_SIZE[1]//4*3)] = headerColor[2]
    header[0:WINDOW_SIZE[0], (WINDOW_SIZE[1]//4*3)
                            :(WINDOW_SIZE[1])] = headerColor[3]
    header = cv2.resize(header, HEADER_SIZE)
    # print(HEADER_SIZE)
    # print(WINDOW_SIZE[0]-HEADER_SIZE[1], WINDOW_SIZE[1]-HEADER_SIZE[0])
    # print(header.shape)

    detector = htm.handDetector(maxHands=1)
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_SIZE[1])
    cap.set(4, WINDOW_SIZE[0])
    xp, yp = 0, 0

    while True:
        success, img = cap.read()
        if success:
            try:
                img = cv2.flip(img, 1)
                hands = detector.findHands(img)
                landmarksList = detector.findPosition(hands)

                if len(landmarksList) != 0:
                    # tip of first and thumb fingers
                    x1, y1 = landmarksList[0][8][1:]
                    x2, y2 = landmarksList[0][4][1:]

                    # second node on thumb finger
                    x3, y3 = landmarksList[0][2][1:]

                    fingers = detector.fingersUp()
                    checkDraw = detector.checkDraw()
                    checkErase = detector.checkErase()
                    # print(fingers)
                    # 1. Handfree Mode - 5 finger up and nothing happends except for moving hands and pointer
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                        xp, yp = 0, 0

                    # 2. Erase Mode - All fingers go down
                    elif checkErase:
                        xp, yp = 0, 0
                        imgCanvas = np.zeros(WINDOW_SIZE, np.uint8)

                    # 3. Thickness Changing Mode - Thumb and first finger go up
                    elif fingers[0] == 0 and fingers[1]:
                        cv2.rectangle(img, (x1, y1 - 25),
                                    (x2, y2 + 25), drawColor, cv2.FILLED)
                        # print(abs(x1-x2)*(abs(y1-25-(y2+25))))
                        calibrator = math.sqrt((x2-x3)**2+(y2-y3)**2)
                        print(calibrator)
                        brushThickness = max(
                            int(abs(x1-x2)*(abs(y1-25-(y2+25)))/(calibrator*15)), 1)
                        print(brushThickness)
                        xp, yp = 0

                    # 4. Selection Color Mode - first and second fingers go up
                    elif fingers[1] and fingers[2]:
                        xp, yp = 0, 0
                        if y1 < HEADER_SIZE[1]:
                            if 0 < x1 < WINDOW_SIZE[1]//4:
                                drawColor = headerColor[0]
                            elif WINDOW_SIZE[1]//4 < x1 < WINDOW_SIZE[1]//2:
                                drawColor = headerColor[1]
                            elif WINDOW_SIZE[1]//2 < x1 < WINDOW_SIZE[1]//4*3:
                                drawColor = headerColor[2]
                            elif WINDOW_SIZE[1]//4*3 < x1 < WINDOW_SIZE[1]:
                                drawColor = headerColor[3]

                    # 5. Drawing Mode - Only first fingers go up
                    elif checkDraw:
                        cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(imgCanvas, (xp, yp), (x1, y1),
                                drawColor, brushThickness)

                        xp, yp = x1, y1

                img = cv2.add(img, imgCanvas)
                # Optionally stack both frames and show it.
                stacked = np.hstack((imgCanvas, img))
                cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.6, fy=0.6))
                img[(WINDOW_SIZE[1]-HEADER_SIZE[0]):HEADER_SIZE[1],
                    (WINDOW_SIZE[1]-HEADER_SIZE[0]):WINDOW_SIZE[1]] = header
                cv2.imshow("Image", img)

            except:
                # print("",end="")
                # print((WINDOW_SIZE[1]-HEADER_SIZE[0]),HEADER_SIZE[1], (WINDOW_SIZE[1]-HEADER_SIZE[0]),WINDOW_SIZE[1])
                img[(WINDOW_SIZE[1]-HEADER_SIZE[0]):HEADER_SIZE[1],
                    (WINDOW_SIZE[1]-HEADER_SIZE[0]):WINDOW_SIZE[1]] = header
                cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k & 0xFF == 27: #ESC to escape
            break
        elif k & 0xFF == 32: #Spacebar to save image
            print(imgCanvas)
            if not os.path.isdir('pics'):
                os.makedirs("pics")
            cv2.imwrite("pics/drawing_{}.png".format(str(datetime.now())
                                                    [:-7].replace(":", "-").replace(" ", "-")), imgCanvas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution",choices=['vga', 'hd'],  help="choosing resolution for webcam", default = "vga")
    args = parser.parse_args()
    if args.resolution == "vga" or args.resolution == "hd":
        main(args.resolution)