import cv2
import numpy as np
import os

def enhancePixel(pixelValue):
    L = 256
    r = pixelValue
    if(L/3 < r < 2*L/3):
        m = (L/6 - 5*L/6) / (2*L/3 - L/3)  # Slope for the decreasing segment
        c = 5*L/6-m*(L/3) # c = y-mx
        s = m *r + c  # y = mx + c
        return int(s)
    else:
        return int(5*L/6)
    
path = "/Users/mean/year-4/image-processing/hw-1/hw1.2"
filename_image = 'traffic.jpg'
filename_image = 'superpets_mini.jpg'
filename_image = 'spellbound_mini.jpg'
name,type = filename_image.split(".")
# Load the input image
image = cv2.imread(filename_image)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_gray = np.array(gray_image)
cv2.imwrite(os.path.join(path,f"{name}_gray.jpg"),image_gray)
enhanced_image = np.zeros_like(image_gray, dtype=np.uint8)
# Apply the custom function to each pixel
for i in range(image_gray.shape[0]):
    for j in range(image_gray.shape[1]):
        enhanced_image[i, j] = enhancePixel(image_gray[i, j])
        # print("old",image_gray[i,j],"new",enhanced_image[i, j])

# cv2.imshow('enhance_image', enhanced_image)
# cv2.waitKey(0)  
# # Window shown waits for any key pressing event
# cv2.destroyAllWindows()

newfilename = f"{name}_enhanced.jpg"
cv2.imwrite(os.path.join(path,newfilename),enhanced_image)

