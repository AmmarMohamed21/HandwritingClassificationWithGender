import cv2
import numpy as np
import math
from PCA import *

def ColdFeature(img):

    #Get edges of the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #To Calculate R and Theta for each point resulted from polygonal approximation algorithm
    RThetaArray=[]

    #loop over contours
    for contour in contours:

        #get the polygonal approximation algorithm
        approxPoly = cv2.approxPolyDP(contour, 1, False)
        approxPoly=approxPoly.reshape(approxPoly.shape[0],2)

        #Calculate R,Theta for each two points
        for i in range(approxPoly.shape[0]-1):
            point = approxPoly[i]
            nextPoint = approxPoly[i+1]
            theta = math.atan2(nextPoint[1]-point[1], nextPoint[0]-point[0])
            radius=np.linalg.norm(nextPoint-point)
            RThetaArray.append([radius,theta])

    RThetaArray = np.array(RThetaArray)

    #remove outliers (Points where distance > 12)
    RThetaArray = RThetaArray[RThetaArray[:,0] <= 12]

    #Transform from polar to XY
    XYArray = []
    for point in RThetaArray:
        XYArray.append([point[0]*math.cos(point[1]),point[0]*math.sin(point[1])])

    XYArray = np.array(XYArray)

    #Calculate 2d Histogram
    NBINS = 10 
    COLD_HIST, xedges,yedges= np.histogram2d(XYArray[:,0],XYArray[:,1], NBINS)
    COLD_HIST = COLD_HIST.flatten().tolist()
    return COLD_HIST