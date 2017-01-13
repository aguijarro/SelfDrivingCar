# Code given by Udacity, complete by Andres Guijarro
# Define a function that takes an image, a color space,
# and a new image size and returns a feature vector

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutout1.jpg')

# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    small_img = cv2.resize(img, size)
    # Use cv2.resize().ravel() to create the feature vector
    features = small_img.ravel()  # Remove this line!
    # Return the feature vector
    return features


def main():

    feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.show()

if __name__ == '__main__':
    main()
