import numpy as np
import cv2 as cv
 
img = cv.imread('input/Image-1.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
 
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
cl1 = clahe.apply(img)
 
cv.imwrite('clahe_2.jpg',cl1)
