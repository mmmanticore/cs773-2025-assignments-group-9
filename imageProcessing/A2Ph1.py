import numpy as np
import cv2
from utilities import *

def main():
    
    # PHASE-2-ASSIGNMENT-1: distortion removal
    
    # Given a chessboard image, you need to undistort the image by the assumption that straight lines in the real world are expected to be straight in the undistorted image.
    # The first step is to estimate kappa (k1) using an iterative algorithm based on RANSAC line fitting.
    # Refer to the lecture and tutorial notes for more details. 
    # Once k1 is estimated you can use "NearestNeighbourInterpolation" function to undistort the original image
    # The provided function "NearestNeighbourInterpolation" needs the distorted image and the estimated kappa as inputs, and it returns the undistorted image. 
    
    img = cv2.imread("W3.png")
    # img = cv2.imread("H3.jpg")
    show(img)
    
    # 1. Import your own Harris corner detection to find all the corners in the chessboard. You will mark down if you use opencv "findChessboardCorners."
    
    # 2. You need to find a way to group the detected corner coordinates on horizontal lines in the chessboard image
    
    # 3. Define the RANSAC line-fitting algorithm to get the best slope and intercept of each horizontal line of points
    
    # 4. Define a function to compute the average error for the fitted lines
    
    # 5. You need an iterative approach to brute force the kappa values
    
    # 6. After estimating kappa, use "NearestNeighbourInterpolation" to undistort the original distorted image.
    
    # PLEASE DEFINE ALL YOUR FUNCTION IN UTILITIES AND IMPORT THEM HERE.
    
    
if __name__ == "__main__":
    main()
    