from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image as im 
import os

def localgammacollection(image, mask,c,gm,k0,k1,k2):
    
    s_k = np.zeros((256,2))
    for i in range(256):
        s_k[i][0] = i/255 #normalize value of pixel is in [0,1]

    for i in range(256):
        s_k[i][1] = (c*(s_k[i][0]**gm))

    max = s_k[255][1]
    for i in range(256):
        s_k[i][1] = s_k[i][1]*255/max

    #create numpy array of image_original
    img_arr = np.array(image, dtype=np.uint8)
    print(img_arr)
    global_mean = np.mean(img_arr)
    global_std = np.std(img_arr)

    h,w = image.shape[0],image.shape[1]
    new_image = np.array(image, dtype=np.uint8)

    global_gammaCollection = np.array(image, dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            global_gammaCollection[x][y] = s_k[img_arr[x][y]][1]
    path = "/Users/mean/year-4/image-processing/hw-2/global-gamma"
    cv2.imwrite(os.path.join(path,f"GPW-w{mask}-c{c}gm{gm}-k0_{k0}:k1_{k1}:k2_{k2}.jpg"),global_gammaCollection)
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
                    
                    new_image[start_x:end_x, start_y:end_y] = np.minimum(255, global_gammaCollection[start_x:end_x, start_y:end_y]).astype(np.uint8)
                else:
                    # Otherwise, copy the original pixel values
                    new_image[start_x:end_x, start_y:end_y] = S_xy
            
    print("done")
    return new_image

image_path = 'Filament.jpg' 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

k0, k1, k2 = 0.4, 0.02, 0.4

k2 = 0.4
#row0
set_k0 = [0.3 ,0.4,0.5] #local
set_k1 = [0.02,0.02,0.02] #local
set_c = [1,1,1]
set_g = [0.4 ,0.4,0.4]
#row1
set_k0 = [0.3 ,0.4,0.5] #local
set_k1 = [0.02,0.02,0.02] #local
set_c = [1,1,1]
set_g = [0.2 ,0.2,0.2]
#row2
set_k0 = [0.3 ,0.3,0.3,0.3] #mean
set_k1 = [0.03 ,0.04,0.05,0.06] #std
set_c = [1 ,1,1,1]
set_g = [0.4 ,0.4,0.4,0.4]
# #row
# set_k0 = [0.03 ,0.035,0.04,0.045]
# set_k1 = [0.02 ,0.02,0.02,0.02]
# set_c = [1 ,1,1,1]
# set_g = [0.2 ,0.2,0.2,0.2]

path = "/Users/mean/year-4/image-processing/hw-2/local-gamma15"
mask = 15
for i in range(len(set_k1)):
    new_img = localgammacollection(image,15,set_c[i], set_g[i],set_k0[i], set_k1[i], k2)
    cv2.imwrite(os.path.join(path,f"GPW-w{mask}-c{set_c[i]}gm{set_g[i]}-k0_{set_k0[i]}:k1_{set_k1[i]}:k2_{k2}.jpg"),new_img)
# new_img_LHE = localHistogramEqualization(image,7,E, k0, k1, k2)
# path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE7"

# cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{k1}-k2:{k2}_2.jpg"),new_img_LHE)