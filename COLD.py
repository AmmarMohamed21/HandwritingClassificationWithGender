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

    #Calculate Average R and Theta
    COLD_AVG_R, COLD_AVG_THETA = np.average(RThetaArray[:,0]), np.average(RThetaArray[:,1])

    #Apply PCA
    X = RThetaArray
    X_norm, mu, sigma = featureNormalize(X)
    U, S = pca(X_norm)
    K = 1
    Z = projectData(X_norm, U, K)
    X_rec = recoverData(Z, U, K)


    #Calculate Average Distance between points and X recovered (The most changes direction)
    diff = X_rec[1] - X_rec[0]
    norm = np.linalg.norm(X_rec[1]-X_rec[0])
    distances = []
    for point in X_norm:
        distance = np.linalg.norm(np.cross(diff, X_rec[0]-point))/norm
        distances.append(distance)

    #Average Distances
    distances = np.array(distances)
    COLD_AVG_DISTANCES = np.average(distances)

    #Slope of PCA
    COLD_PCA_SLOPE = math.atan2(X_rec[1][1]-X_rec[0][1], X_rec[1][0]-X_rec[0][0])

    return COLD_AVG_R, COLD_AVG_THETA, COLD_AVG_DISTANCES, COLD_PCA_SLOPE