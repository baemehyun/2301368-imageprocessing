import cv2
import numpy as np
import os

def power_raw_tranfromation(image,c,g):
    
    enhancedimage = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = image[i, j]
            nom_r = r/255
            s = c*(nom_r**g)
            
            s_new = int(min(s*255,255))
            enhancedimage[i, j] = int(s_new)
    return enhancedimage


filename_image = 'superpets_mini.jpg'
# filename_image = 'traffic.jpg'
# filename_image = 'galaxy3.jpeg'
name,type = filename_image.split(".")

# Load the input image
image = cv2.imread(filename_image)
path = "/Users/mean/year-4/image-processing/hw-1/hw1.3"

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = np.array(gray_image)
cv2.imwrite(os.path.join(path,f"{name}_gray.jpg"),image_gray)
# print(image_gray)

const_arr = [0.5,1,2]
gamma_arr = [0.4,2.5]

#combination of const and gamma
for c in range(len(const_arr)):
    for g in range(len(gamma_arr)):
        newfilename = f"{name}enhanced_with_c{str(const_arr[c])}g{str(gamma_arr[g])}.jpg"
        enhanced_image = power_raw_tranfromation(image_gray,const_arr[c],gamma_arr[g])
        cv2.imwrite(os.path.join(path,newfilename),enhanced_image)
        