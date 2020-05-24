import numpy as np
import cv2
import imutils
from imutils import contours
import matplotlib.pyplot as plt
from skimage import measure


def get_car_plate(img: np.ndarray, h: int=80, w: int=400):
    img = cv2.resize(img,(620,480))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 255)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = np.array([])

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt != np.array([]):
        pts = screenCnt.tolist()
        up = sorted(pts)[:2]
        down = sorted(pts)[2:]
        up = sorted(up, key=lambda x: x[0][1])
        down = sorted(down, key=lambda x: x[0][1])
        pts = np.array([*up, *down], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts, np.array([[0,0], [0,h], [w,0], [w,h]], dtype=np.float32))
        warp = cv2.warpPerspective(img, M, (w, h))

        return True, warp
    else:
        return False, np.array([])


def get_car_plate_signs(car_plate_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    signs = []

    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if h > 40 and w > 30:
            roi = thresh[y-5:y + h+5, x-5:x + w+5]
            print(roi)
            # cv2.imshow('Car plate', car_plate_img)
            # cv2.imshow('Car plate sign', roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            #roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
            #roi = cv2.dilate(roi, np.ones((3,3), np.uint8), iterations=2)
            signs.append(roi)

    return signs


def perform_processing(image: np.ndarray) -> str:
    print(image.shape)
    detection, car_plate_img = get_car_plate(image) 

    if detection:
        signs = get_car_plate_signs(car_plate_img)
        print(len(signs))
        # for sign in signs:
        #     cv2.imshow('Car plate', car_plate_img)
        #     cv2.imshow('Car plate sign', sign)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
    else:
        print('Car plate not detected')

    # TODO: add image processing here
    return 'PO12345'
