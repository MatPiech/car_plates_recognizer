import cv2
import numpy as np

class ExtractPlate:
    def __init__(self):
        pass

    def maximizeContrast(self, img_gray: np.ndarray) -> np.ndarray:
        """Function to maximize contrast of processed image.

        Parameters
        ----------
        img_gray : np.ndarray
            Image in grayscale.

        Returns
        -------
        np.ndarray
            Image with maximized contrast.
        """
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        img_TopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structuring_element)
        img_BlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structuring_element)

        img_gray_plus_TopHat = cv2.add(img_gray, img_TopHat)
        img_gray_plus_TopHat_minus_BlackHat = cv2.subtract(img_gray_plus_TopHat, img_BlackHat)

        return img_gray_plus_TopHat_minus_BlackHat

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Function preprocess image to extract car plate.

        Parameters
        ----------
        img : np.ndarray
            Color image.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_value = cv2.split(hsv)[2]

        img_max_contrast_gray = self.maximizeContrast(img_value)

        img_blurred = cv2.GaussianBlur(img_max_contrast_gray, (5, 5), 0)

        img_thresh = cv2.adaptiveThreshold(img_blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        return img_thresh

    def findpossible_characteracters(self, img_thresh: np.ndarray) -> list:
        """Function finds possible car plate characters.

        Parameters
        ----------
        img_thresh : np.ndarray
            Preprocessed and thresholded image.

        Returns
        -------
        list
            List with possible plate characters localization in form (x,y,w,h)
        """
        list_of_possible_characteracters = []

        cnts, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            ratio = w / h 
        
            if (area > 80 and w > 2 and h > 8 and 0.25 < ratio and ratio < 1):
                list_of_possible_characteracters.append((x,y,w,h))

        return list_of_possible_characteracters

    def findMatchingChars(self, possible_character: tuple, list_of_characters: list) -> list:
        """Function to check if possible_character meet the conditions of car plate character.

        Parameters
        ----------
        possible_character : tuple
            Tested contour of possible character in form (x,y,w,h).
        list_of_characters : list
            List of all possible characters.

        Returns
        -------
        list
            Regions which meet the conditions of possible character.
        """
        list_of_matching_characters = []
        x, y, w, h = possible_character
        area = w * h
        
        for possible_matching_characters in list_of_characters:
            if possible_matching_characters == possible_character:
                continue

            x_i,y_i,w_i,h_i = possible_matching_characters
            area_i = w_i * h_i
            diagonal = np.sqrt((w_i ** 2) + (h_i ** 2))
            
            distance_x = abs((2*x+w)/2 - (2*x_i+w_i)/2)
            distance_y = abs((2*y+h)/2 - (2*y_i+h_i)/2)
                
            distance_between_chars = np.sqrt((distance_x ** 2) + (distance_y ** 2))
            
            if distance_x != 0.0:
                angle_in_rad = np.arctan(distance_y / distance_x)
            else:
                angle_in_rad = 1.5708

            angle_between_chars = angle_in_rad * (180.0 / np.pi)

            change_in_area = float(abs(area - area_i)) / float(area)

            change_in_width = float(abs(w - w_i)) / float(w)
            change_in_height = float(abs(h - h_i)) / float(h)

            if (distance_between_chars < (diagonal * 5.0) and angle_between_chars < 12.0 and
                change_in_area < 0.5 and change_in_width < 0.8 and change_in_height < 0.2):

                list_of_matching_characters.append(possible_matching_characters)

        return list_of_matching_characters

    def findListOfListsOfMatchingChars(self, list_of_possible_characters: list) -> list:
        """Function which generate all possible plates.

        Parameters
        ----------
        list_of_possible_characters : list
            List of all possible characters.

        Returns
        -------
        list
            List of lists which contain car plate charcters needed to crop plate.
        """
        list_of_lists_of_matching_chars = []

        for possible_character in list_of_possible_characters:
            list_of_matching_characters = self.findMatchingChars(possible_character, list_of_possible_characters)

            list_of_matching_characters.append(possible_character)

            if len(list_of_matching_characters) < 3:
                continue

            list_of_lists_of_matching_chars.append(list_of_matching_characters)

            list_of_possible_characters_with_current_matches_removed = []

            list_of_possible_characters_with_current_matches_removed = list(set(list_of_possible_characters) - set(list_of_matching_characters))

            recursivelist_of_lists_of_matching_chars = self.findListOfListsOfMatchingChars(list_of_possible_characters_with_current_matches_removed)

            for recursivelist_of_matching_characters in recursivelist_of_lists_of_matching_chars:
                list_of_lists_of_matching_chars.append(recursivelist_of_matching_characters)

            break

        return list_of_lists_of_matching_chars

    def extractPlate(self, img: np.ndarray, list_of_matching_characters: list) -> np.ndarray:
        """Function extracts region which possibly contain car plate.

        Parameters
        ----------
        img : np.ndarray
            Original image of car's front or back.
        list_of_matching_characters : list
            [description]

        Returns
        -------
        np.ndarray
            Extracted region with car plate.
        """
        list_of_matching_characters.sort(key=lambda x: x[0])
        
        x_0, y_0, w_0, h_0 = list_of_matching_characters[0]
        x_n, y_n, w_n, h_n = list_of_matching_characters[-1]

        center_x_0 = (2*x_0+w_0) / 2
        center_x_n = (2*x_n+w_n) / 2
        center_y_0 = (2*y_0+h_0) / 2
        center_y_n = (2*y_n+h_n) / 2
            
        plate_center_x = (center_x_0 + center_x_n) / 2.0
        plate_center_y = (center_y_0 + center_y_n) / 2.0

        plate_center = (plate_center_x, plate_center_y)

        plate_width = int((x_n + w_n - x_0) * 1.3)

        total_of_char_heights = 0

        for matchingChar in list_of_matching_characters:
            total_of_char_heights += matchingChar[3]

        average_char_height = total_of_char_heights / len(list_of_matching_characters)

        plate_height = int(average_char_height * 1.5)

        opposite = center_y_n - center_y_0
        hypotenuse = np.sqrt(((center_x_0-center_x_n) ** 2) + ((center_y_0-center_y_n) ** 2))
        correction_angle_in_rad = np.arcsin(opposite / hypotenuse)
        correction_angle_in_deg = correction_angle_in_rad * (180.0 / np.pi)

        rotation_matrix = cv2.getRotationMatrix2D(plate_center, correction_angle_in_deg, 1.0)

        height, width = img.shape[:2]

        img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

        img_cropped = cv2.getRectSubPix(img_rotated, (plate_width, plate_height), plate_center)
        
        return img_cropped

    def detect(self, img: np.ndarray) -> list:
        """Main function of ExtractPlate class 

        Parameters
        ----------
        img : np.ndarray
            Original image of car's front or back.

        Returns
        -------
        list
            Extracted possible plates.
        """
        list_of_possible_plates = []
        
        img_thresh = self.preprocess(img)
        
        list_of_possible_characteracters = self.findpossible_characteracters(img_thresh)
        
        list_of_lists_of_matching_characters = self.findListOfListsOfMatchingChars(list_of_possible_characteracters)
        
        for list_of_matching_characters in list_of_lists_of_matching_characters:
            img_cropped = self.extractPlate(img, list_of_matching_characters)

            if img_cropped is not None:
                list_of_possible_plates.append(img_cropped)
        
        return list_of_possible_plates