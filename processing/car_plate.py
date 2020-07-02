import numpy as np
import cv2
import imutils
from imutils import contours
from itertools import product

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
        self.plate_second_part_roi = np.load(
            'data/plate_second_part_signs.npy')

    def getCarPlateSigns(self, car_plate_img: np.ndarray, threshold: int = 100, th2: int = 0):
        """Function to find car plate numbers and read them.

        Parameters
        ----------
        car_plate_img : np.ndarray
            Image with car plate.
        threshold : int, optional
            Threshold value, by default 100.
        th2 : int, optional
            Additional variable for threshold start when in first iteration car plate is not detected, by default 0.
        """
        gray = cv2.cvtColor(car_plate_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY_INV)

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            cnts = contours.sort_contours(cnts)[0]

        self.numbers = []
        roi_h_size = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 40 and w > 5 and w < 60:
                roi = thresh[y-np.min([y, 5]): y+h+np.min([thresh.shape[0]-y-h, 5]),
                             x-np.min([x, 5]): x+w+np.min([thresh.shape[1]-x-w, 5])]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones(
                        (3, 3), np.uint8), iterations=1)
                    if len(self.numbers) < 2:
                        num = self.recognizeSign(
                            roi, self.plate_first_part_roi, self.plate_first_part_signs)
                    else:
                        num = self.recognizeSign(
                            roi, self.plate_second_part_roi, self.plate_second_part_signs)

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
        img_copy = image.copy()
        EP = ExtractPlate()

        possible_car_plate_signs_list = []

        approx_poly_epsilon = [0.02, 0.014]
        scale = [0.5, 0.3]

        config_combinations = list(
            product(*[approx_poly_epsilon, scale]))

        for epsilon, s in config_combinations:
            car_plate_imgs = EP.detectPlateSimple(
                img=image, scale=s, epsilon=epsilon)

            if len(car_plate_imgs) == 0:
                continue

            for plate in car_plate_imgs:
                self.getCarPlateSigns(plate, threshold=100)
                car_plate_signs = "".join(self.numbers)

                if car_plate_signs != "" and len(car_plate_signs) <= 7:
                    possible_car_plate_signs_list.append(car_plate_signs)

                if len(car_plate_signs) == 7:
                    return car_plate_signs

        for s in [0.25, 0.5]:
            img = cv2.resize(img_copy, None, fx=s, fy=s)
            car_plate_imgs = EP.detectPlate(img)
            if len(car_plate_imgs) > 0:
                for plate in car_plate_imgs[:10]:
                    plate = cv2.resize(plate, (400, 80))
                    self.getCarPlateSigns(plate, threshold=100)
                    car_plate_signs = "".join(self.numbers)

                    if car_plate_signs != "" and len(car_plate_signs) <= 7:
                        possible_car_plate_signs_list.append(car_plate_signs)

                    if len(car_plate_signs) == 7:
                        return car_plate_signs

        if len(possible_car_plate_signs_list) == 0:
            return "???????"
        else:
            return sorted(possible_car_plate_signs_list, key=lambda x: len(x))[-1]
