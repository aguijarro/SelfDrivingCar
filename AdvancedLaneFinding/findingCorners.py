import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    # prepare object points
    nx = 8#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    # Make a list of calibration images
    fname = './calibration_wide/GOPR0058.jpg'
    img = cv2.imread(fname)
    plt.imshow(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
