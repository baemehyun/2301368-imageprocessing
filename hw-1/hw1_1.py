import cv2
import numpy as np
import PIL 
import os

def resize_pixel_replication(image, factor,filename):
    h,w = image.shape
    new_h = int(h*factor)
    new_w = int(w*factor)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(f'resized_{filename}_img', resized_image)
    cv2.waitKey(0)  
    # Window shown waits for any key pressing event
    cv2.destroyAllWindows()
    
    np.savetxt(os.path.join(path,f'after_{filename}.txt'), image_np, fmt='%d', delimiter=',')
    print(f"Array saved to 'after_{filename}.txt'")
    print('success')
    cv2.imwrite(os.path.join(path,filename), resized_image) # Save the image
   
def resize_bilinear_interpolation(image,factor,filename):
    h,w = image.shape
    new_h = int(h*factor)
    new_w = int(w*factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    cv2.imshow(f'resized_{filename}_img', resized_image)
    cv2.waitKey(0)  
    # Window shown waits for any key pressing event
    cv2.destroyAllWindows()
    
    np.savetxt(os.path.join(path,f'after_{filename}.txt'), image_np, fmt='%d', delimiter=',')
    print(f"Array saved to 'after_{filename}.txt'")
    print('success')
    # path = "/Users/mean/year-4/image-processing/hw-1/hw1.1"
    cv2.imwrite(os.path.join(path,filename), resized_image) # Save the image
    
path = "/Users/mean/year-4/image-processing/hw-1/hw1.1"
# filename_image = 'flower.jpg'
filename_image = 'fractal.jpg'
# filename_image = 'traffic.jpg'
name,type = filename_image.split(".")
# Load the input image
image = cv2.imread(filename_image)
# cv2.imshow('Original', image)
# cv2.waitKey(0)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Grayscale', gray_image)
# cv2.waitKey(0)  
# Window shown waits for any key pressing event
# cv2.destroyAllWindows()

h,w = gray_image.shape
image_np = np.array(gray_image)
cv2.imwrite(os.path.join(path,f"{name}gray.jpg"), image_np) # Save the image
np.savetxt(os.path.join(path,f'before_{name}.txt'), image_np, fmt='%d', delimiter=',')
print(f"Array saved to 'before_{name}.txt'")

# zoom image to 3x
resize_pixel_replication(gray_image,3,f"{name}_zoomed_replication.jpg")
resize_bilinear_interpolation(gray_image,3,f"{name}_zoomed_bilinear.jpg")

# shurnk image to 1/3x
resize_pixel_replication(gray_image,1/3,f"{name}_shrunk_replication.jpg")
resize_bilinear_interpolation(gray_image,1/3,f"{name}_shrunk_bilinear.jpg")
