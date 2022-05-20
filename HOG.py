from skimage.feature import hog
import numpy as np


def calculateHOG(img):

	fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
						cells_per_block=(2, 2), visualize=True, multichannel=False)
	return np.average(hog_image)