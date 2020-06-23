import numpy as np
import cv2
import imutils
from imutils import contours
import matplotlib.pyplot as plt                     # TO REMOVE
from itertools import product                       # TO REMOVE
import time                                         # TO REMOVE

from processing.extract_plate import ExtractPlate


class CarPlate:
    """Class with simple car plate extraction method and plate's nummber reading functions.
    """
    def __init__(self):
        self.plate_first_part_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                       'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.plate_second_part_signs = ['A', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
                                        'U', 'V', 'W', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.readTemplateSigns()


    def readTemplateSigns(self):
        """Loads signs templates.
        """
        self.plate_first_part_roi = np.load('data/plate_first_part_signs.npy')
        self.plate_second_part_roi = np.load('data/plate_second_part_signs.npy')


    def getCarPlate(self, img: np.ndarray, d: int=11, th1: int=30, epsilon: float=0.02, scale: float=0.5, h: int=80, w: int=400) -> list:
        """Function to extract possible car plates.

        Parameters
        ----------
        img : np.ndarray
            Analyzed image of car front or back with license plate.
        d : int, optional
            Diameter of bilateral filter, by default 11
        th1 : int, optional
            Lower threshold of canny edge detector, by default 30
        epsilon : float, optional
            Coefficient for accuracy approximation in approxPolyDP function used with length of contour, by default 0.02
        scale : float, optional
            Scale of analyzed image in comparisson to original image, by default 0.5
        h : int, optional
            Possible plate height, by default 80
        w : int, optional
            Possible plate width, by default 400

        Returns
        -------
        list
            Possible plates.
        """
        img = cv2.resize(img,(int(img.shape[0]*scale),int(img.shape[1]*scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d, 17, 17)

        edged = cv2.Canny(gray, th1, 255)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]  
        screenCnts = []

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon * peri, True)
            if len(approx) == 4:
                screenCnts.append(approx)

        warps = []
        for screenCnt in screenCnts:
            if screenCnt != np.array([]):
                pts = screenCnt.tolist()
                up = sorted(pts)[:2]
                down = sorted(pts)[2:]
                up = sorted(up, key=lambda x: x[0][1])
                down = sorted(down, key=lambda x: x[0][1])
                pts = np.array([up[0][0], up[1][0], np.add(down[0][0], [0,5]), np.add(down[1][0], [0,5])], dtype=np.float32)

                M = cv2.getPerspectiveTransform(pts, np.array([[0,0], [0,h], [w,0], [w,h]], dtype=np.float32))
                warp = cv2.warpPerspective(img, M, (w, h))
                warps.append(warp)

            return warps
        else:
            return []

    
    def getCarPlateSigns(self, car_plate_img: np.ndarray, threshold: int=100, th2: int=0):
        """Function to find car plate numbers and read them.

        Parameters
        ----------
        car_plate_img : np.ndarray
            Image with car plate.
        threshold : int, optional
            Threshold value, by default 100
        """
        gray = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            cnts = contours.sort_contours(cnts)[0]

        self.numbers = []
        roi_h_size = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 40 and w > 5 and w < 60:
                roi = thresh[y-np.min([y, 5]): y+h+np.min([thresh.shape[0]-y-h, 5]), x-np.min([x, 5]): x+w+np.min([thresh.shape[1]-x-w, 5])]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
                    if len(self.numbers) < 2:
                        num = self.recognizeSign(roi, self.plate_first_part_roi, self.plate_first_part_signs)
                    else:
                        num = self.recognizeSign(roi, self.plate_second_part_roi, self.plate_second_part_signs)

                    if w < 20 and (num != 'I' and num != 'J' and num != '1') or num == '':
                        continue 
                    self.numbers.append(num)
                    roi_h_size.append(roi.shape[0])
                
        while len(roi_h_size) > 7:
            self.numbers.pop(roi_h_size.index(min(roi_h_size)))
            roi_h_size.remove(min(roi_h_size))

        if len(self.numbers) < 7 and threshold <= 150:
            self.getCarPlateSigns(car_plate_img, th2+25, th2+25)

    
    def recognizeSign(self, sign_roi: np.ndarray, reference_signs_roi: np.ndarray, reference_signs: list) -> str:
        """Function to recognize sign by matching it with template.

        Parameters
        ----------
        sign_roi : np.ndarray
            Image part with possible sign.
        reference_signs_roi : np.ndarray
            Array with signs templates.
        reference_signs : list
            List with signs in defined combination.

        Returns
        -------
        str
            Recognized sign.
        """
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
        """Main function in car plate characters recognition task.

        Parameters
        ----------
        image : np.ndarray
            Image with front or back of car.

        Returns
        -------
        str
            Recognized car plate characters.
        """
        start = time.time()
        img_copy = image.copy()
        EP = ExtractPlate()

        possible_car_plate_signs_list = []

        canny_threshold1 = [10]
        approx_poly_epsilon = [0.02, 0.014]
        scale = [0.5, 0.3]

        config_combinations = list(product(*[canny_threshold1, approx_poly_epsilon, scale]))

        for th1, epsilon, s in config_combinations:
            car_plate_imgs = self.getCarPlate(img=image, scale=s, th1=th1, epsilon=epsilon)

            if (time.time() - start) > 1.25:
                break

            if len(car_plate_imgs) == 0:
                continue

            for plate in car_plate_imgs:
                self.getCarPlateSigns(plate, threshold=100)
                car_plate_signs = "".join(self.numbers)
            
                if car_plate_signs != "" and len(car_plate_signs) <= 7:
                    possible_car_plate_signs_list.append(car_plate_signs)

                if len(car_plate_signs) == 7:
                    print(time.time() - start)                      # TO REMOVE
                    return car_plate_signs

        for s in [0.25, 0.5]:
            if (time.time() - start) > 1.25:
                break
            img = cv2.resize(img_copy, None, fx=s, fy=s)
            car_plate_imgs = EP.detect(img)
            if len(car_plate_imgs) > 0:
                for plate in car_plate_imgs[:10]:
                    plate = cv2.resize(plate, (400,80))
                    if (time.time() - start) > 1.25:
                        break
                    self.getCarPlateSigns(plate, threshold=100)
                    car_plate_signs = "".join(self.numbers)
                
                    if car_plate_signs != "" and len(car_plate_signs) <= 7:
                        possible_car_plate_signs_list.append(car_plate_signs)

                    if len(car_plate_signs) == 7:
                        print(time.time() - start)                  # TO REMOVE
                        return car_plate_signs

        print(possible_car_plate_signs_list)                        # TO REMOVE
        print(time.time() - start)                                  # TO REMOVE

        if len(possible_car_plate_signs_list) == 0:
            return "???????"
        else:
            return sorted(possible_car_plate_signs_list, key=lambda x: len(x))[-1]
