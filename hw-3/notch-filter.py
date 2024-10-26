import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
def spat_domain_to_freq_domain(img):
    f = np.fft.fft2(img)
    fshiftaf = np.fft.fftshift(f)
    magnitude_spectrum_af = np.log(np.abs(fshiftaf)+1)
    return magnitude_spectrum_af
def create_notch_filter(r,img):
    h,w=img.shape[0],img.shape[1]
    mask_high = np.zeros(img.shape,dtype=int)
    mask_low = np.zeros(img.shape,dtype=int)
    x = h/2 #h
    y = w/2 #k
    for i in range(h):
        for j in range(w):
            if(pow(pow(i-x,2)+pow(j-y,2),0.5)<=r):
                mask_low[i][j] = 1
            if(pow(pow(i-x,2)+pow(j-y,2),0.5)>r):
                mask_high[i][j] = 1
    return mask_high,mask_low

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
            mask_low = H_uv
            mask_high = 1-H_uv
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
    print(magnitude_spectrum)
    # img_back = cv2.normalize(dft_filtered, None, 0, 255, cv2.NORM_MINMAX)
    return G_uv,magnitude_spectrum
    
    
#Filtering with notch filter
img = cv2.imread('Noisy_Tom_Jerry.jpg')
# img = cv2.imread('image.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
freq_dm = spat_domain_to_freq_domain(gray_image)
notch_filters = create_notch_filter(100,gray_image)
result_low_pass,r1 = filtering(freq_dm,notch_filters[1],gray_image)
result_high_pass,r2 = filtering(freq_dm,notch_filters[0],gray_image)

plt.figure(figsize=(12, 8))
plt.subplot(421),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(422),plt.imshow(freq_dm, cmap = 'gray')
plt.title('F(u,v)'), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(notch_filters[1], cmap = 'gray')
plt.title('H(u,v) Low pass'), plt.xticks([]), plt.yticks([])

plt.subplot(424),plt.imshow(notch_filters[0], cmap = 'gray')
plt.title('H(u,v) High pass'), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(result_low_pass, cmap = 'gray')
plt.title('G(u,v) Low pass'), plt.xticks([]), plt.yticks([])

plt.subplot(426),plt.imshow(result_high_pass, cmap = 'gray')
plt.title('G(u,v) High pass'), plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(427),plt.imshow(r1, cmap = 'gray')
plt.title('result Low pass'), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(r2, cmap = 'gray')
plt.title('result High pass'), plt.xticks([]), plt.yticks([])
plt.show()

