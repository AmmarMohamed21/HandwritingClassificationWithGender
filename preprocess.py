from skimage import io, morphology
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.filters import threshold_yen
import cv2

import os

def ExtractLines(imgPath, img,targetDirectory,isPredict=False):
    img=img*255
    img=img.astype('uint8')

    #Finding line contours
    kernel = np.ones((3, 50), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index=1

    linesList = []

    #Crop line contours
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        if w < 270 or h<35 or h>60:
            continue



        #cropping the line
        line = img[y :y +  h , x : x + w]

        approxPoly = cv2.approxPolyDP(contour, 1, False)
        approxPoly=approxPoly.reshape(approxPoly.shape[0],2)

        if (not isPredict):
            s = 'image' + 'crop_' + str(index) + '.png'
            cv2.imwrite(targetDirectory+imgPath+s , line)
        else:
            linesList.append(line)
        index = index + 1
    return linesList

def PreprocessImage(imgPath, index=1, targetDirectory='', isPrediction=False):
    img = io.imread(imgPath)
    #img = img [50:img.shape[0]-100, 50:img.shape[1]-50]
    #resize image
    targetSize = (1240,800)
    resizedimg = resize(img,targetSize)

    #rgp to gray
    grayimg = rgb2gray(resizedimg)

    #gray to binary
    threshold = threshold_yen(grayimg)
    binaryImg = np.where(grayimg>threshold, 1,0)
    #apply opening on image
    openinigImage = morphology.opening(binaryImg)

    #Remove large black areas
    footprint=np.ones((10,10))
    close = morphology.binary_closing(openinigImage, footprint)
    inv = np.invert(close)
    result=openinigImage+inv
    
    #Extract image lines
    linesList = ExtractLines(str(index), result,targetDirectory,isPredict=isPrediction)
    
    return linesList