# Modify the values of the variables red_threshold, green_threshold,
# and blue_threshold until you are able to retain as much of the lane
# lines as possible, while getting rid of most of the other stuff.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def ColorSelector():
    # Read in the image and print out some stats
    image = (mpimg.imread('test.png') * 255).astype('uint8')
    print('This image is: ', type(image),
          'with dimensions:', image.shape)

    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    # Define color selection criteria
    # MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    print('Esta es la variable rgb_threshold: ', rgb_threshold)

    # Do a bitwise or with the "|" character to identify
    # pixels below the thresholds
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                  | (image[:, :, 1] < rgb_threshold[1]) \
                  | (image[:, :, 2] < rgb_threshold[2])

    print('Esta es la variable thresholds: ', thresholds)

    color_select[thresholds] = [0, 0, 0]
    # plt.imshow(color_select)

    # Uncomment the following code if you are running the code
    # locally and wish to save the image
    mpimg.imsave("test-after.png", color_select)

    # Display the image
    plt.imshow(color_select)
    plt.show()

if __name__ == '__main__':
    ColorSelector()
