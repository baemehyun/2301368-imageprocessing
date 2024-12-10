import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('input/Image-1.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian for fine details
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Compute the gradient using Sobel filters
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient = cv2.magnitude(sobel_x, sobel_y)

# Smooth the gradient with Gaussian blur
smoothed_gradient = cv2.GaussianBlur(gradient, (5, 5), 0)

# Normalize the smoothed gradient to use as a mask
mask = cv2.normalize(smoothed_gradient, None, 0, 1, cv2.NORM_MINMAX)

# Mask the Laplacian using the smoothed gradient
masked_laplacian = cv2.multiply(np.abs(laplacian), mask)

# Apply a gray-level transformation (contrast stretching)
enhanced_image = cv2.normalize(masked_laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 4, 2)
plt.title("Laplacian")
plt.imshow(np.abs(laplacian), cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Smoothed Gradient")
plt.imshow(smoothed_gradient, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Enhanced Image")
plt.imshow(enhanced_image, cmap='gray')

plt.show()
