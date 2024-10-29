import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
def spat_domain_to_freq_domain(img):
    f = np.fft.fft2(img)
    fshiftaf = np.fft.fftshift(f)
    magnitude_spectrum_af = np.log(np.abs(fshiftaf)+1)
    return magnitude_spectrum_af

def create_filter_re(D_0,img):
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

def create_bandreject_filter(img,D_0,W):
    #find hieght, width
    row = img.shape[0]
    col = img.shape[1]
    bandRejectFilter = np.zeros(img.shape,dtype=int)
    bandPassFilter = np.zeros(img.shape ,dtype=int)
    #assume x,y is center point of image
    x = col/2
    y = row/2
    for u in range(row):
        for v in range(col):
            distance = pow(pow(u-y,2)+pow(v-x,2),0.5)
            if(distance<D_0-(W/2)):
                bandRejectFilter[u][v] = 1
                bandPassFilter[u][v] = 0
            elif(D_0-(W/2)<=distance<=D_0+(W/2)):
                bandRejectFilter[u][v] = 0
                bandPassFilter[u][v] = 1
            else:
                bandRejectFilter[u][v]=1
                bandPassFilter[u][v] = 0
    
    return bandRejectFilter,bandPassFilter
    
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
    
    
#Filtering with notch filter
img = cv2.imread('Noisy_Tom_Jerry.jpg')
# img = cv2.imread('image.png')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
freq_dm = spat_domain_to_freq_domain(gray_image)
bandReject, bandPass = create_bandreject_filter(gray_image,11,2)
result_low_pass,r1 = filtering(freq_dm,bandReject,gray_image)
result_high_pass,r2 = filtering(freq_dm,bandPass,gray_image)
nameOfFilter = 'ideal'


plt.figure(figsize=(12, 8))
plt.subplot(421),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(422),plt.imshow(freq_dm, cmap = 'gray')
plt.title('F(u,v)'), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(bandReject, cmap = 'gray')
plt.title(f'H(u,v) {nameOfFilter} Bandreject'), plt.xticks([]), plt.yticks([])

plt.subplot(424),plt.imshow(bandPass, cmap = 'gray')
plt.title(f'H(u,v) {nameOfFilter} Bandpass'), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(result_low_pass, cmap = 'gray')
plt.title('G(u,v) Bandreject'), plt.xticks([]), plt.yticks([])

plt.subplot(426),plt.imshow(result_high_pass, cmap = 'gray')
plt.title('G(u,v) Bandpass'), plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(427),plt.imshow(r1, cmap = 'gray')
plt.title('result Bandreject'), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(r2, cmap = 'gray')
plt.title('result Bandpass'), plt.xticks([]), plt.yticks([])
plt.show()
