import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
image = cv2.imread('input/Image-1.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply minimum filter using erosion
kernel = np.ones((5,5), np.uint8)
min_filter = cv2.erode(image, kernel)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Minimum Filtered Image')
plt.imshow(min_filter, cmap='gray')

plt.show()
