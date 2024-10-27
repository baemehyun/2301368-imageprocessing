import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
def spat_domain_to_freq_domain(img):
    f = np.fft.fft2(img)
    fshiftaf = np.fft.fftshift(f)
    magnitude_spectrum_af = np.log(np.abs(fshiftaf)+1)
    return magnitude_spectrum_af

def create_gaussian_filter(D_0,img):
    h,w=img.shape[0],img.shape[1]
    mask_high = np.zeros(img.shape,dtype=int)
    mask_low = np.zeros(img.shape,dtype=int)
    origin_x = h/2
    origin_y = w/2
    for i in range(h):
        for j in range(w):
            #calc distance
            D_uv = pow(pow(origin_x-i,2)+pow(origin_y-j,2),0.5)
            top = -pow(D_uv,2)/(2*pow(D_0,2))
            H_uv = pow(math.e,top)
            mask_low[i][j] = H_uv*255
            mask_high[i][j] = (1-H_uv)*255
            
    return mask_high,mask_low
    
def filtering(freq,mask,img):
    G_uv = freq*mask
    #DFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # print(fshift)
    enhance = mask*fshift #G(u,v)=H(u,v)*F(u,v)

    #IDFT
    # Compute the inverse Fourier Transform
    shift_if = np.fft.ifftshift(enhance) 
    inverse_f = np.fft.ifft2(shift_if) #g(x,y)
    
    magnitude_spectrum = np.abs(inverse_f)
    # print(magnitude_spectrum)
    # img_back = cv2.normalize(dft_filtered, None, 0, 255, cv2.NORM_MINMAX)
    return G_uv,magnitude_spectrum
    
def normalize_and_convert(image):
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
    
#Filtering with notch filter
img = cv2.imread('Noisy_Tom_Jerry.jpg')
img = cv2.imread('Noisy_whirlpool.jpg')
img = cv2.imread('Noisy_galaxy3.jpg')
# img = cv2.imread('image.png')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
freq_dm = spat_domain_to_freq_domain(gray_image)
d0 = 50
gaussian_filters = create_gaussian_filter(d0,gray_image)
result_low_pass,r1 = filtering(freq_dm,gaussian_filters[1],gray_image)
result_high_pass,r2 = filtering(freq_dm,gaussian_filters[0],gray_image)

#save the result
path = "/Users/mean/year-4/image-processing/2301368-imageprocessing/hw-3/result-of-galaxy"
cv2.imwrite(os.path.join(path,f"gray.jpg"),gray_image)
cv2.imwrite(os.path.join(path,f"freq.jpg"),normalize_and_convert(freq_dm))
# plt.imsave(os.path.join(path, "freqNonNorm.png"), freq_dm, cmap='gray')
cv2.imwrite(os.path.join(path,f"freqOfLowpass-gaus-{d0}.jpg"),normalize_and_convert(result_low_pass))
cv2.imwrite(os.path.join(path,f"freqOfHighpass-gaus-{d0}.jpg"),normalize_and_convert(result_high_pass))
# plt.imsave(os.path.join(path, "freqOfLowpassNonNorm.png"), result_low_pass, cmap='gray')
cv2.imwrite(os.path.join(path,f"enhancedLowpass-gaus-{d0}.jpg"),normalize_and_convert(r1))
cv2.imwrite(os.path.join(path,f"enhancedHighpass-gaus-{d0}.jpg"),normalize_and_convert(r2))

# plt.figure(figsize=(12, 8))
# plt.subplot(421),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(422),plt.imshow(freq_dm, cmap = 'gray')
# plt.title('F(u,v)'), plt.xticks([]), plt.yticks([])

# plt.subplot(423),plt.imshow(gaussian_filters[1], cmap = 'gray')
# plt.title('H(u,v) Low pass'), plt.xticks([]), plt.yticks([])

# plt.subplot(424),plt.imshow(gaussian_filters[0], cmap = 'gray')
# plt.title('H(u,v) High pass'), plt.xticks([]), plt.yticks([])

# plt.subplot(425),plt.imshow(result_low_pass, cmap = 'gray')
# plt.title('G(u,v) Low pass'), plt.xticks([]), plt.yticks([])

# plt.subplot(426),plt.imshow(result_high_pass, cmap = 'gray')
# plt.title('G(u,v) High pass'), plt.xticks([]), plt.yticks([])
# # plt.show()

# plt.subplot(427),plt.imshow(r1, cmap = 'gray')
# plt.title('result Low pass'), plt.xticks([]), plt.yticks([])

# plt.subplot(428),plt.imshow(r2, cmap = 'gray')
# plt.title('result High pass'), plt.xticks([]), plt.yticks([])
# plt.show()


