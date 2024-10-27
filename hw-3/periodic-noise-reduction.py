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

def notch_bandreject_filter(img,D_0,u0,v0):
    #find hieght, width
    M = img.shape[0]
    N = img.shape[1]
    bandRejectFilter = np.zeros(img.shape,dtype=int)
    bandPassFilter = np.zeros(img.shape ,dtype=int)
    #assume x,y is center point of image
    x = M/2
    y = N/2
    for u in range(M):
        for v in range(N):
            distance1 = pow(pow(u-x-u0,2)+pow(v-y-v0,2),0.5)
            distance2 = pow(pow(u-x+u0,2)+pow(v-y+v0,2),0.5)
            if(distance1<=D_0 or distance2<=D_0):
                bandRejectFilter[u][v] = 0
                bandPassFilter[u][v] = 1
            else:
                bandRejectFilter[u][v] = 1
                bandPassFilter[u][v] = 0

    return bandRejectFilter,bandPassFilter

def notch_bandreject_extend(img,bR,bP,D_0,u0,v0):
    #find hieght, width
    M = img.shape[0]
    N = img.shape[1]
    bandRejectFilter = bR
    bandPassFilter = bP
    #assume x,y is center point of image
    x = M/2
    y = N/2
    for u in range(M):
        for v in range(N):
            distance1 = pow(pow(u-x-u0,2)+pow(v-y-v0,2),0.5)
            distance2 = pow(pow(u-x+u0,2)+pow(v-y+v0,2),0.5)
            if(distance1<=D_0 or distance2<=D_0):
                bandRejectFilter[u][v] = 0
                bandPassFilter[u][v] = 1

    return bandRejectFilter,bandPassFilter
  
def guassian_bandreject_filter(img,D_0,u0,v0):
    #find hieght, width
    M = img.shape[0]
    N = img.shape[1]
    bandRejectFilter = np.zeros(img.shape,dtype=int)
    bandPassFilter = np.zeros(img.shape ,dtype=int)
    #assume x,y is center point of image
    x = M/2
    y = N/2
    for u in range(M):
        for v in range(N):
            distance1 = pow(pow(u-x-u0,2)+pow(v-y-v0,2),0.5)
            distance2 = pow(pow(u-x+u0,2)+pow(v-y+v0,2),0.5)
            bandRejectFilter[u][v] = (1-math.exp(-0.5*(distance1*distance2/D_0)))*255
            bandPassFilter[u][v] = 255-bandRejectFilter[u][v]
            # if(distance1<=D_0 or distance2<=D_0):
            #     bandRejectFilter[u][v] = 0
            #     bandPassFilter[u][v] = 1
            # else:
            #     bandRejectFilter[u][v] = 1
            #     bandPassFilter[u][v] = 0

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
    
def normalize_and_convert(image):
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
#Filtering with notch filter
# img = cv2.imread('Noisy_whirlpool.jpg')
img = cv2.imread('Noisy_galaxy3.jpg')
# img = cv2.imread('image.png')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
freq_dm = spat_domain_to_freq_domain(gray_image)
d0,u0,v0=4,0,17
bandReject, bandPass = notch_bandreject_filter(gray_image,d0,u0,v0)
d0,u0,v0=4,0,277
bandReject, bandPass = notch_bandreject_extend(gray_image,bandReject,bandPass,d0,u0,v0)
d0,u0,v0=3,0,243
bandReject, bandPass = notch_bandreject_extend(gray_image,bandReject,bandPass,d0,u0,v0)
# d0,u0,v0=3,7,6
# bandReject, bandPass = notch_bandreject_extend(gray_image,bandReject,bandPass,d0,u0,v0)

# bandReject, bandPass = guassian_bandreject_filter(gray_image,7,8,0)
result_low_pass,r1 = filtering(freq_dm,bandReject,gray_image)
result_high_pass,r2 = filtering(freq_dm,bandPass,gray_image)
nameF = 'notch'
nameOfFilter = 'notch'

#save the result
path = "/Users/mean/year-4/image-processing/2301368-imageprocessing/hw-3/result-of-galaxy/test"
cv2.imwrite(os.path.join(path,f"gray.jpg"),gray_image)
cv2.imwrite(os.path.join(path,f"freq.jpg"),normalize_and_convert(freq_dm))
cv2.imwrite(os.path.join(path,f"freqOfbandReject-{nameF}-D{d0}u{u0}v{v0}.jpg"),normalize_and_convert(result_low_pass))
cv2.imwrite(os.path.join(path,f"freqOfbandPass-{nameF}-D{d0}u{u0}v{v0}.jpg"),normalize_and_convert(result_high_pass))
cv2.imwrite(os.path.join(path,f"enhancedbandReject-{nameF}-D{d0}u{u0}v{v0}.jpg"),normalize_and_convert(r1))
cv2.imwrite(os.path.join(path,f"enhancedbandPass-{nameF}-D{d0}u{u0}v{v0}.jpg"),normalize_and_convert(r2))

# plt.figure(figsize=(15, 8))
# plt.subplot(421),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(212),plt.imshow(freq_dm, cmap = 'gray')
# plt.title('F(u,v)'), plt.xticks([]), plt.yticks([])
# plt.show()
# plt.subplot(423),plt.imshow(bandReject, cmap = 'gray')
# plt.title(f'H(u,v) {nameOfFilter} Bandreject'), plt.xticks([]), plt.yticks([])

# plt.subplot(424),plt.imshow(bandPass, cmap = 'gray')
# plt.title(f'H(u,v) {nameOfFilter} Bandpass'), plt.xticks([]), plt.yticks([])

# plt.subplot(425),plt.imshow(result_low_pass, cmap = 'gray')
# plt.title('G(u,v) Bandreject'), plt.xticks([]), plt.yticks([])

# plt.subplot(426),plt.imshow(result_high_pass, cmap = 'gray')
# plt.title('G(u,v) Bandpass'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.subplot(427),plt.imshow(r1, cmap = 'gray')
# plt.title('result Bandreject'), plt.xticks([]), plt.yticks([])

# plt.subplot(428),plt.imshow(r2, cmap = 'gray')
# plt.title('result Bandpass'), plt.xticks([]), plt.yticks([])
# plt.show()
