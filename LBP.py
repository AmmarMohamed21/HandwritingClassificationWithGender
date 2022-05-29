# <====== //// ====== ----- Import libraries ----- ====== //// ======>
from turtle import width
import cv2
import numpy as np
from matplotlib import pyplot as plt

# <====== //// ====== ----- Threshold Pixels Function ----- ====== //// ======>
"""
function: threshold_pixels
parameters: image, center pixel, position (x, y)
usage: it returns 1 if value of given pixel position > value of center pixel 
note: we use (try, except) to avoid errors such as (index is out of bounds for axis 0)
"""
def threshold_pixels(img, cen_pixel, x, y):
    val = 0
    try:
        if img[x][y] >= cen_pixel:
            val = 1
    except:
        pass
    return val

# <====== //// ====== ----- Calculate LBP Function ----- ====== //// ======>
"""
function: calculate_lbp
parameters: image, position (x, y)
usage: it calculates LBP for given pixel. 
"""
def calculate_lbp(img, x, y):
    cen = img[x][y]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    thr_pixels = []
    
    # ==== pass eight pixels to threshold_pixels function
    thr_pixels.append(threshold_pixels(img, cen, x-1, y+1))     # (Top, Right) pixel
    thr_pixels.append(threshold_pixels(img, cen, x, y+1))       # (Right) pixel
    thr_pixels.append(threshold_pixels(img, cen, x+1, y+1))     # (Bottom, Right) pixel
    thr_pixels.append(threshold_pixels(img, cen, x+1, y))       # (Bottom) pixel
    thr_pixels.append(threshold_pixels(img, cen, x+1, y-1))     # (Bottom, Left) pixel
    thr_pixels.append(threshold_pixels(img, cen, x, y-1))       # (Left) pixel
    thr_pixels.append(threshold_pixels(img, cen, x-1, y-1))     # (Top, Left) pixel
    thr_pixels.append(threshold_pixels(img, cen, x-1, y))       # (Top) pixel
    
    val = 0
    for i in range(len(thr_pixels)):
        val += thr_pixels[i] * power_val[i]
    return val    


# <====== //// ====== ----- Extract LBP feature from image Function ----- ====== //// ======>
"""
function: extract_lbp
parameters: image
usage: it calculates LBP for given image. 
"""
def extract_lbp(gray_img):
    height =gray_img.shape[0]
    width = gray_img.shape[1]
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = calculate_lbp(gray_img, i, j)
    
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256]).flatten().tolist()
    return hist_lbp