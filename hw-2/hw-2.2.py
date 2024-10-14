from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image as im 
import os

def localHistogramEqualization(image, mask,E,k0, k1, k2):
    #create table of histogram equalization
    # compute a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    s_k = np.zeros((256,2))
    for i in range(256):
        s_k[i][0] = i #this value is defined for gray level 
        
    hist_arr_int = np.array(hist,dtype=int)
    hist_sum = 0
    for i in range(256):
        hist_sum += hist_arr_int[i][0]

    sum_sk =0
    for i in range(256):
        nk_n = hist_arr_int[i][0]/hist_sum
        sum_sk = sum_sk+round(nk_n,2)
        s_k[i][1] = sum_sk
        
    for i in range(256):
        s_k[i][1] = min(int(s_k[i][1]*255),255)
    
    #create numpy array of image_original
    img_arr = np.array(image, dtype=np.uint8)
    print(img_arr)
    global_mean = np.mean(img_arr)
    global_std = np.std(img_arr)
    # mask =3
    step =int((mask-1)/2)
    h,w = image.shape[0],image.shape[1]
    # w = 426
    # h = 499
    check_outside = []
    
    new_image = np.array(image, dtype=np.uint8)
    max_arr = np.full((mask, mask), 255)
    
    for x in range(0, h, mask):  # Move step by step based on mask size
            for y in range(0, w, mask):
                # Get the local window centered around the pixel (x, y)
                start_x, end_x = x, min(x + mask, h)
                start_y, end_y = y, min(y + mask, w)
                S_xy = img_arr[start_x:end_x, start_y:end_y]
                local_mean = np.mean(S_xy)
                local_std = np.std(S_xy)
                
                # Check condition for local enhancement
                if local_mean <= k0 * global_mean and k1 * global_std <= local_std <= k2 * global_std:
                    # Apply enhancement to the entire local area
                    print(x,y,img_arr[x][y]*4)
                    print("enhance","local_mean",local_mean,"global_mean",global_mean,"local_std",local_std,"global_std",global_std)
                    new_image[start_x:end_x, start_y:end_y] = np.minimum(255, S_xy * E).astype(np.uint8)
                else:
                    # Otherwise, copy the original pixel values
                    new_image[start_x:end_x, start_y:end_y] = S_xy
            
    print("done")
    return new_image

image_path = 'Filament.jpg' 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

E, k0, k1, k2 =4, 0.4, 0.02, 0.4
# E, k0, k1, k2 =4, 0.3, 0.02, 0.4
# E, k0, k1, k2 =4, 0.5, 0.02, 0.4
# E, k0, k1, k2 =4, 0.3, 0.015, 0.4
# E, k0, k1, k2 =4, 0.3, 0.025, 0.4
# E, k0, k1, k2 =4, 0.3, 0.03, 0.4
# E, k0, k1, k2 =4, 0.3, 0.035, 0.4
# E, k0, k1, k2 =4, 0.4, 0.025, 0.4
# E, k0, k1, k2 =4, 0.4, 0.03, 0.4
# E, k0, k1, k2 =4, 0.4, 0.035, 0.4
# E, k0, k1, k2 =4, 0.4, 0.04, 0.4
# E, k0, k1, k2 =4, 0.4, 0.045, 0.4
# E, k0, k1, k2 =4, 0.35, 0.045, 0.4
# E, k0, k1, k2 =4, 0.4, 0.05, 0.4

#size 7x7
# E, k0, k1, k2 =4, 0.4, 0.02, 0.4
# E, k0, k1, k2 =4, 0.3, 0.02, 0.4
# E, k0, k1, k2 =4, 0.5, 0.02, 0.4
# E, k0, k1, k2 =4, 0.35, 0.02, 0.4
# E, k0, k1, k2 =4, 0.35, 0.025, 0.4
# E, k0, k1, k2 =4, 0.35, 0.03, 0.4
# E, k0, k1, k2 =4, 0.35, 0.035, 0.4
# E, k0, k1, k2 =4, 0.35, 0.04, 0.4
# E, k0, k1, k2 =4, 0.4, 0.025, 0.4
# E, k0, k1, k2 =4, 0.4, 0.03, 0.4
# E, k0, k1, k2 =4, 0.4, 0.035, 0.4
# E, k0, k1, k2 =4, 0.4, 0.04, 0.4
# E, k0, k1, k2 =4, 0.45, 0.025, 0.4
# E, k0, k1, k2 =4, 0.45, 0.03, 0.4
# E, k0, k1, k2 =4, 0.45, 0.035, 0.4
# E, k0, k1, k2 =4, 0.45, 0.04, 0.4

# E, k0, k1, k2 =4, 0.3, 0.04, 0.4
# set_k1 = [0.025,0.03,0.035,0.04]

# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE7"
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE3"

# for i in range(len(set_k1)):
#     new_img_LHE = localHistogramEqualization(image,3,E, k0, set_k1[i], k2)
#     cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{set_k1[i]}-k2:{k2}_2.jpg"),new_img_LHE)

# E,  k2 =4,  0.4
# set_k0 = [0.4,0.4,0.35]
# set_k1 = [0.045,0.05,0.045]
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE7"

# for i in range(len(set_k1)):
#     new_img_LHE = localHistogramEqualization(image,7,E, set_k0[i], set_k1[i], k2)
#     cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{set_k0[i]}-k1:{set_k1[i]}-k2:{k2}_2.jpg"),new_img_LHE)

#size 11x11
# E, k1, k2 =4, 0.02 ,0.4
# set_k0 = [0.3,0.35,0.4,0.5]
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE11"

# for i in range(len(set_k0)):
#     new_img_LHE = localHistogramEqualization(image,11,E, set_k0[i], k1, k2)
#     cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{set_k0[i]}-k1:{k1}-k2:{k2}_2.jpg"),new_img_LHE)
    
# E, k0, k2 =4, 0.035 ,0.4
# set_k1 = [0.025 ,0.03,0.035,0.04]
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE11"

# for i in range(len(set_k1)):
#     new_img_LHE = localHistogramEqualization(image,11,E, k0, set_k1[i], k2)
#     cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{set_k1[i]}-k2:{k2}_2.jpg"),new_img_LHE)

# E, k0, k2 =4, 0.04 ,0.4
# set_k1 = [0.025 ,0.03,0.035,0.04]
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE11"

# for i in range(len(set_k1)):
#     new_img_LHE = localHistogramEqualization(image,11,E, k0, set_k1[i], k2)
#     cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{set_k1[i]}-k2:{k2}_2.jpg"),new_img_LHE)

E, k2 =4, 0.4
#row0
set_k0 = [0.3 ,0.35,0.4,0.5]
set_k1 = [0.02 ,0.02,0.02,0.02]
#row1
set_k0 = [0.35 ,0.35,0.35,0.35]
set_k1 = [0.025 ,0.03,0.035,0.04]
#row2
set_k0 = [0.45 ,0.45,0.45,0.45]
set_k1 = [0.025 ,0.03,0.035,0.04]
# #row3
set_k0 = [0.45 ,0.45,0.35]
set_k1 = [0.045 ,0.05,0.045]

path = "/Users/mean/year-4/image-processing/hw-2/test2_local_HE11"

for i in range(len(set_k1)):
    new_img_LHE = localHistogramEqualization(image,11,E, set_k0[i], set_k1[i], k2)
    cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{set_k0[i]}-k1:{set_k1[i]}-k2:{k2}_2.jpg"),new_img_LHE)
# new_img_LHE = localHistogramEqualization(image,7,E, k0, k1, k2)
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE7"

# cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{k1}-k2:{k2}_2.jpg"),new_img_LHE)