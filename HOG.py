from skimage.feature import hog
import numpy as np
import cv2

def calculateHOG(img):

	resized_img = np.resize(img, (30, 300))
	hog_hist = hog(resized_img, orientations=9, pixels_per_cell=(10, 100),
                	cells_per_block=(2, 2), visualize=False, multichannel=False)

	return hog_hist