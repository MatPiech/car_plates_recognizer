import numpy as np
import cv2
import imutils
from imutils import contours
import matplotlib.pyplot as plt


class CarPlate:
    def __init__(self):
        self.plate_first_part_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                       'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.plate_second_part_signs = ['A', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
                                        'U', 'V', 'W', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.readTemplateSigns()


    def readTemplateSigns(self):
        self.plate_first_part_roi = np.load('data/plate_first_part_signs.npy')
        self.plate_second_part_roi = np.load('data/plate_second_part_signs.npy')


    def getCarPlate(self, img: np.ndarray, h: int=80, w: int=400):
        img = cv2.resize(img,(620,480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        edged = cv2.Canny(gray, 30, 255)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = np.array([])

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt != np.array([]):
            pts = screenCnt.tolist()
            up = sorted(pts)[:2]
            down = sorted(pts)[2:]
            up = sorted(up, key=lambda x: x[0][1])
            down = sorted(down, key=lambda x: x[0][1])
            #pts = np.array([up[0][0], up[1][0], np.subtract(down[0][0], [0,2]), np.subtract(down[1][0], [0,2])], dtype=np.float32)
            pts = np.array([up[0][0], up[1][0], np.add(down[0][0], [0,5]), np.add(down[1][0], [0,5])], dtype=np.float32)

            M = cv2.getPerspectiveTransform(pts, np.array([[0,0], [0,h], [w,0], [w,h]], dtype=np.float32))
            warp = cv2.warpPerspective(img, M, (w, h))

            return True, warp
        else:
            return False, np.array([])

    
    def getCarPlateSigns(self, car_plate_img: np.ndarray, threshold: int=100) -> np.ndarray:
        gray = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        self.signs = []
        h_mean = [cv2.boundingRect(c)[3] for c in cnts]

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if h > (np.mean(h_mean)-5) and h > 40 and w > 20 and w < 60:
                roi = thresh[y-np.min([y, 5]): y+h+np.min([thresh.shape[0]-y-h, 5]), x-np.min([x, 5]): x+w+np.min([thresh.shape[1]-x-w, 5])]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    #roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
                    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
                    #roi = cv2.dilate(roi, np.ones((3,3), np.uint8), iterations=2)
                    self.signs.append(roi)

        if self.signs == [] and threshold<250:
            self.getCarPlateSigns(car_plate_img, threshold+25)

    
    def recognizeSign(self, sign_roi: np.ndarray, reference_signs_roi: np.ndarray, reference_signs: list) -> str:
        sign_roi = cv2.resize(sign_roi, (50, 75))
        scores = []
        
        for reference_roi in reference_signs_roi:
            result = cv2.matchTemplate(sign_roi, reference_roi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        
        if max(scores) > 10000000:
            return reference_signs[int(np.argmax(scores))]
        else:
            return ''


    def process(self, image: np.ndarray) -> str:
        detection_found, car_plate_img = self.getCarPlate(img=image)

        if detection_found:
            self.getCarPlateSigns(car_plate_img)
            first_two_letters = list(map(lambda roi: self.recognizeSign(roi, self.plate_first_part_roi, self.plate_first_part_signs), self.signs[:2]))
            other_letters = list(map(lambda roi: self.recognizeSign(roi, self.plate_second_part_roi, self.plate_second_part_signs), self.signs[2:]))
            #print(np.mean([sign.shape for sign in self.signs], axis=0))
            #print(first_two_letters, other_letters)
            car_plate_signs = "".join([*first_two_letters, *other_letters])
        else:
            car_plate_signs = "XXXXXXX"

        print(car_plate_signs)
        return car_plate_signs
