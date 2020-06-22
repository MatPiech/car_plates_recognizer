import cv2
import numpy as np

class ExtractPlate:
    def __init__(self):
        pass

    def extractValue(self, img: np.ndarray) -> np.ndarray:
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        imgValue = cv2.split(imgHSV)[2]

        return imgValue

    def maximizeContrast(self, gray: np.ndarray) -> np.ndarray:
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        imgGrayscale = self.extractValue(img)

        imgMaxContrastGrayscale = self.maximizeContrast(imgGrayscale)

        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5, 5), 0)

        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        return imgGrayscale, imgThresh

    def findPossibleCharsInScene(self, imgThresh):
        listOfPossibleChars = []                # this will be the return value

        cnts, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w * h
            ratio = w / h 
        
            if (area > 80 and w > 2 and h > 8 and 0.25 < ratio and ratio < 1):
                listOfPossibleChars.append((x,y,w,h))

        return listOfPossibleChars

    def findListOfMatchingChars(self, possibleChar, listOfChars):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
        listOfMatchingChars = []                # this will be the return value
        x,y,w,h = possibleChar
        area = w * h
        
        for possibleMatchingChar in listOfChars:                # for each char in big list
            if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                                                        # then we should not include it in the list of matches b/c that would end up double including the current char
                continue                                # so do not add to list of matches and jump back to top of for loop
            # end if
            x1,y1,w1,h1 = possibleMatchingChar
            area1 = w1 * h1
            fltDiagonalSize = np.sqrt((w1 ** 2) + (h1 ** 2))
                        # compute stuff to see if chars are a match
            X = abs((2*x+w)/2 - (2*x1+w1)/2)
            Y = abs((2*y+h)/2 - (2*y1+h1)/2)
                
            fltDistanceBetweenChars = np.sqrt((X ** 2) + (Y ** 2))
            
            if X != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
                fltAngleInRad = np.arctan(Y / X)      # if adjacent is not zero, calculate angle
            else:
                fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
            # end if

            fltAngleBetweenChars = fltAngleInRad * (180.0 / np.pi)       # calculate angle in degrees

            fltChangeInArea = float(abs(area - area1)) / float(area)

            fltChangeInWidth = float(abs(w - w1)) / float(w)
            fltChangeInHeight = float(abs(h - h1)) / float(h)

                    # check if chars match
            if (fltDistanceBetweenChars < (fltDiagonalSize * 5.0) and fltAngleBetweenChars < 12.0 and
                fltChangeInArea < 0.5 and fltChangeInWidth < 0.8 and fltChangeInHeight < 0.2):

                listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars

        return listOfMatchingChars                  # return result

    def findListOfListsOfMatchingChars(self, listOfPossibleChars):
        listOfListsOfMatchingChars = []                  # this will be the return value

        for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
            listOfMatchingChars = self.findListOfMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char

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

    def extractPlate(self, imgOriginal, listOfMatchingChars):

        #listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position
        listOfMatchingChars.sort(key=lambda x: x[0])
        
                # calculate the center point of the plate
        x0, y0, w0, h0 = listOfMatchingChars[0]
        xn, yn, wn, hn = listOfMatchingChars[-1]
        X0 = (2*x0+w0) / 2
        Xn = (2*xn+wn) / 2
        Y0 = (2*y0+h0) / 2
        Yn = (2*yn+hn) / 2
            
        fltPlateCenterX = (X0 + Xn) / 2.0
        fltPlateCenterY = (Y0 + Yn) / 2.0

        ptPlateCenter = fltPlateCenterX, fltPlateCenterY

                # calculate plate width and height
        intPlateWidth = int((xn + wn - x0) * 1.3)

        intTotalOfCharHeights = 0

        for matchingChar in listOfMatchingChars:
            intTotalOfCharHeights = intTotalOfCharHeights + matchingChar[3]

        fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

        intPlateHeight = int(fltAverageCharHeight * 1.5)

                # calculate correction angle of plate region
        fltOpposite = Yn - Y0
        fltHypotenuse = np.sqrt(((X0-Xn) ** 2) + ((Y0-Yn) ** 2))
        fltCorrectionAngleInRad = np.arcsin(fltOpposite / fltHypotenuse)
        fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / np.pi)

                # final steps are to perform the actual rotation

                # get the rotation matrix for our calculated correction angle
        rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

        height, width, numChannels = imgOriginal.shape      # unpack original image width and height

        imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

        imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
        
        return imgCropped

    def detectPlatesInScene(self, img):
        listOfPossiblePlates = []
        
        height, width, numChannels = img.shape

        imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
        imgThreshScene = np.zeros((height, width, 1), np.uint8)
        imgContours = np.zeros((height, width, 3), np.uint8)
        
        imgGrayscaleScene, imgThreshScene = self.preprocess(img)
        
        listOfPossibleCharsInScene = self.findPossibleCharsInScene(imgThreshScene)
        
        listOfListsOfMatchingCharsInScene = self.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
        
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
            imgCropped = self.extractPlate(img, listOfMatchingChars)         # attempt to extract plate

            if imgCropped is not None:                          # if plate was found
                listOfPossiblePlates.append(imgCropped)                  # add to list of possible plates
        
        return listOfPossiblePlates