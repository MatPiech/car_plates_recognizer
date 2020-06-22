import numpy as np
import cv2
import imutils
from imutils import contours
import matplotlib.pyplot as plt
from itertools import product
import time

from processing.extract_plate import ExtractPlate


class CarPlate:
    def __init__(self):
        self.plate_first_part_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                       'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.plate_second_part_signs = ['A', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
                                        'U', 'V', 'W', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.readTemplateSigns()
        self.numbers = []


    def readTemplateSigns(self):
        self.plate_first_part_roi = np.load('data/plate_first_part_signs.npy')
        self.plate_second_part_roi = np.load('data/plate_second_part_signs.npy')


    def getCarPlate(self, img: np.ndarray, d: int=11, th1: int=30, epsilon: float=0.02, scale=0.5, h: int=80, w: int=400):
        img = cv2.resize(img,(int(img.shape[0]*scale),int(img.shape[1]*scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d, 17, 17)

        edged = cv2.Canny(gray, th1, 255)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = np.array([])

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon * peri, True)
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

            return True, [warp]
        else:
            return False, np.array([])

    
    def getCarPlateSigns(self, car_plate_img: np.ndarray, threshold: int=100) -> np.ndarray:
        gray = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            cnts = contours.sort_contours(cnts)[0]
        self.signs = []
        self.numbers = []
        h_mean = [cv2.boundingRect(c)[3] for c in cnts]

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if h > (np.mean(h_mean)-5) and h > 40 and w > 5 and w < 60:
                roi = thresh[y-np.min([y, 5]): y+h+np.min([thresh.shape[0]-y-h, 5]), x-np.min([x, 5]): x+w+np.min([thresh.shape[1]-x-w, 5])]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    #roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
                    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
                    #roi = cv2.dilate(roi, np.ones((3,3), np.uint8), iterations=2)
                    if len(self.numbers) < 2:
                        num = self.recognizeSign(roi, self.plate_first_part_roi, self.plate_first_part_signs)
                    else:
                        num = self.recognizeSign(roi, self.plate_second_part_roi, self.plate_second_part_signs)

                    if w < 20 and (num != 'I' and num != 'J' and num != '1') or num == '':
                        continue 
                    self.numbers.append(num)
                    self.signs.append(roi)
                if len(self.numbers) == 7:
                    break

        if self.numbers == [] and threshold<200:
            self.getCarPlateSigns(car_plate_img, threshold+50)

    
    def recognizeSign(self, sign_roi: np.ndarray, reference_signs_roi: np.ndarray, reference_signs: list) -> str:
        sign_roi = cv2.resize(sign_roi, (50, 75))
        scores = []
        
        for reference_roi in reference_signs_roi:
            result = cv2.matchTemplate(sign_roi, reference_roi, cv2.TM_CCOEFF)
            #result = cv2.matchTemplate(sign_roi, reference_roi, cv2.TM_CCOEFF_NORMED)
            
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        
        if max(scores) > 10000000:
            return reference_signs[int(np.argmax(scores))]
        else:
            return ''


    def process(self, image: np.ndarray) -> str:
        img_copy = image.copy()
        start = time.time()
        EP = ExtractPlate()
        possible_car_plate_signs_list = []

        #bilateral_filter_diameter = [11, 9, 7]
        canny_threshold1 = [10] #, 50, 80]
        approx_poly_epsilon = [0.02, 0.014] #, 0.002]
        scale = [0.5, 0.3]

        config_combinations = list(product(*[canny_threshold1, approx_poly_epsilon, scale]))

        for th1, epsilon, s in config_combinations:
            detection_found, car_plate_imgs = self.getCarPlate(img=image, scale=s, th1=th1, epsilon=epsilon)

            if (time.time() - start) > 1.25:
                break
            if not detection_found:
                continue

            for plate in car_plate_imgs:
                self.getCarPlateSigns(plate)
                car_plate_signs = "".join(self.numbers)
            
                if car_plate_signs != "" and len(car_plate_signs) <= 7:
                    possible_car_plate_signs_list.append(car_plate_signs)

                if len(car_plate_signs) == 7:
                    print(time.time() - start)
                    return car_plate_signs

            #if list(filter(lambda x: x == 7, car_plate_signs)):
            #    break

        for s in [0.25, 0.5]:
            img = cv2.resize(img_copy, None, fx=s, fy=s)
            car_plate_imgs = EP.detectPlatesInScene(img)
            if len(car_plate_imgs) > 0:
                for plate in car_plate_imgs:
                    plate = cv2.resize(plate, (400,80))
                    if (time.time() - start) > 1.15:
                        break
                    self.getCarPlateSigns(plate)
                    car_plate_signs = "".join(self.numbers)
                
                    if car_plate_signs != "" and len(car_plate_signs) <= 7:
                        possible_car_plate_signs_list.append(car_plate_signs)

                    if len(car_plate_signs) == 7:
                        print(time.time() - start)
                        return car_plate_signs

        print(possible_car_plate_signs_list)
        print(time.time() - start)

        if len(possible_car_plate_signs_list) == 0:
            return "???????"
        else:
            return sorted(possible_car_plate_signs_list, key=lambda x: len(x))[-1]
