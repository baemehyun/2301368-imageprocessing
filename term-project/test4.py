import cv2 as cv
import numpy as np

# Load the image in grayscale
img = cv.imread('input/Image-1.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

# Get the image dimensions
height, width = img.shape

# Split the image into top and bottom halves
top_half = img[:height//2, :]
bottom_half = img[height//2:, :]

# Define the enhancement method
enhancement_method = "local"  # Change to "global" for global histogram equalization

if enhancement_method == "local":
    # Apply CLAHE for local enhancement
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    top_enhanced = clahe.apply(top_half)
    bottom_enhanced = clahe.apply(bottom_half)
else:
    # Apply global histogram equalization
    top_enhanced = cv.equalizeHist(top_half)
    bottom_enhanced = cv.equalizeHist(bottom_half)

# Combine the halves back into one image
enhanced_img = np.vstack((top_enhanced, bottom_enhanced))

# Save the result
cv.imwrite('enhanced_image.jpg', enhanced_img)
cv.imshow("Enhanced Image", enhanced_img)
cv.waitKey(0)
cv.destroyAllWindows()
