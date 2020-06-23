import cv2
import numpy as np

class ExtractPlate:
    def __init__(self):
        pass

    def maximizeContrast(self, gray: np.ndarray) -> np.ndarray:
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

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

        imgValue = cv2.split(hsv)[2]

        maxContrastGray = self.maximizeContrast(imgValue)

        blurred = cv2.GaussianBlur(maxContrastGray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        return thresh

    def findPossibleChars(self, thresh: np.ndarray) -> list:
        """Function finds possible car plate characters.

        Parameters
        ----------
        thresh : np.ndarray
            Preprocessed and thresholded image.

        Returns
        -------
        list
            List with possible plate characters localization in form (x,y,w,h)
        """
        listOfPossibleChars = []

        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w * h
            ratio = w / h 
        
            if (area > 80 and w > 2 and h > 8 and 0.25 < ratio and ratio < 1):
                listOfPossibleChars.append((x,y,w,h))

        return listOfPossibleChars

    def findMatchingChars(self, possibleChar, listOfChars) -> list:
        """[summary]

        Parameters
        ----------
        possibleChar : [type]
            [description]
        listOfChars : [type]
            [description]

        Returns
        -------
        list
            [description]
        """
        listOfMatchingChars = []
        x,y,w,h = possibleChar
        area = w * h
        
        for possibleMatchingChar in listOfChars:
            if possibleMatchingChar == possibleChar:
                continue

            xi,yi,wi,hi = possibleMatchingChar
            areai = wi * hi
            diagonal = np.sqrt((wi ** 2) + (hi ** 2))
            
            distanceX = abs((2*x+w)/2 - (2*xi+wi)/2)
            distanceY = abs((2*y+h)/2 - (2*yi+hi)/2)
                
            distanceBetweenChars = np.sqrt((distanceX ** 2) + (distanceY ** 2))
            
            if distanceX != 0.0:
                angleInRad = np.arctan(distanceY / distanceX)
            else:
                angleInRad = 1.5708

            angleBetweenChars = angleInRad * (180.0 / np.pi)

            changeInArea = float(abs(area - areai)) / float(area)

            changeInWidth = float(abs(w - wi)) / float(w)
            changeInHeight = float(abs(h - hi)) / float(h)

            if (distanceBetweenChars < (diagonal * 5.0) and angleBetweenChars < 12.0 and
                changeInArea < 0.5 and changeInWidth < 0.8 and changeInHeight < 0.2):

                listOfMatchingChars.append(possibleMatchingChar)

        return listOfMatchingChars

    def findListOfListsOfMatchingChars(self, listOfPossibleChars: list) -> list:
        """[summary]

        Parameters
        ----------
        listOfPossibleChars : list
            [description]

        Returns
        -------
        list
            [description]
        """
        listOfListsOfMatchingChars = []                  # this will be the return value

        for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
            listOfMatchingChars = self.findMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char

            listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars

            if len(listOfMatchingChars) < 3:     # if current possible list of matching chars is not long enough to constitute a possible plate
                continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                    # to save the list in any way since it did not have enough chars to be a possible plate
            # end if

                                                    # if we get here, the current list passed test as a "group" or "cluster" of matching chars
            listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

            listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                    # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                    # make sure to make a new big list for this since we don't want to change the original big list
            listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

            recursiveListOfListsOfMatchingChars = self.findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
            # end for

            break

        return listOfListsOfMatchingChars

    def extractPlate(self, img: np.ndarray, listOfMatchingChars: list) -> np.ndarray:
        """Function extracts region which possibly contain car plate.

        Parameters
        ----------
        img : np.ndarray
            Original image of car's front or back.
        listOfMatchingChars : list
            [description]

        Returns
        -------
        np.ndarray
            Extracted region with car plate.
        """
        listOfMatchingChars.sort(key=lambda x: x[0])
        
        x0, y0, w0, h0 = listOfMatchingChars[0]
        xn, yn, wn, hn = listOfMatchingChars[-1]
        centerX0 = (2*x0+w0) / 2
        centerXn = (2*xn+wn) / 2
        centerY0 = (2*y0+h0) / 2
        centerYn = (2*yn+hn) / 2
            
        plateCenterX = (centerX0 + centerXn) / 2.0
        plateCenterY = (centerY0 + centerYn) / 2.0

        plateCenter = (plateCenterX, plateCenterY)

        plateWidth = int((xn + wn - x0) * 1.3)

        totalOfCharHeights = 0

        for matchingChar in listOfMatchingChars:
            totalOfCharHeights += matchingChar[3]

        fltAverageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

        plateHeight = int(fltAverageCharHeight * 1.5)

                # calculate correction angle of plate region
        fltOpposite = centerYn - centerY0
        fltHypotenuse = np.sqrt(((centerX0-centerXn) ** 2) + ((centerY0-centerYn) ** 2))
        fltCorrectionAngleInRad = np.arcsin(fltOpposite / fltHypotenuse)
        fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / np.pi)

                # final steps are to perform the actual rotation

                # get the rotation matrix for our calculated correction angle
        rotationMatrix = cv2.getRotationMatrix2D(plateCenter, fltCorrectionAngleInDeg, 1.0)

        height, width = img.shape[:2]

        imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

        imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), plateCenter)
        
        return imgCropped

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
        listOfPossiblePlates = []
        
        thresh = self.preprocess(img)
        
        listOfPossibleChars = self.findPossibleChars(thresh)
        
        listOfListsOfMatchingCharsInScene = self.findListOfListsOfMatchingChars(listOfPossibleChars)
        
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            imgCropped = self.extractPlate(img, listOfMatchingChars)

            if imgCropped is not None:
                listOfPossiblePlates.append(imgCropped)
        
        return listOfPossiblePlates