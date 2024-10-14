from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image as im 
import os


# matplotlib expects RGB images so convert and then display the image
# with matplotlib
# plt.figure()
# plt.axis("off")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))



# # plot the histogram
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()

# normalize the histogram
# hist /= hist.sum()
# print(hist)

# # plot the normalized histogram
# plt.figure()
# plt.title("Grayscale Histogram (Normalized)")
# plt.xlabel("Bins")
# plt.ylabel("% of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()
def globalHE(image):
    # compute a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    s_k = np.zeros((256,2))
    for i in range(256):
        s_k[i][0] = i
        
    hist_arr_int = np.array(hist,dtype=int)
    hist_sum = 0
    for i in range(256):
        # print(hist_arr_int[i][0])
        hist_sum += hist_arr_int[i][0]

    sum_sk =0
    for i in range(256):
        nk_n = hist_arr_int[i][0]/hist_sum
        sum_sk = sum_sk+round(nk_n,2)
        s_k[i][1] = sum_sk
        # print(s_k[i][1])

    for i in range(256):
        s_k[i][1] = round(s_k[i][1]*255)
    # print(s_k)

    img = np.array(image)
    img_new = np.zeros((image.shape[0],image.shape[1]))
    # print(image.shape)
    for x in range(len(img)):
        for y in range(len(img[0])):
            img_new[x][y] = s_k[img[x][y]][1]
    return img_new
# print(img)
def localHE(image,mask_size):
    # compute a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    r_k = np.zeros((256,2))
    img_arr = np.array(image)
    for i in range(256):
        r_k[i][0] = i
        
    hist_arr_int = np.array(hist,dtype=int)
    hist_sum = 0
    for i in range(256):
        # print(hist_arr_int[i][0])
        hist_sum += hist_arr_int[i][0]
    #normalize p(r) at r_k[i][1]
    for i in range(256):
        nk_n = hist_arr_int[i][0]/hist_sum
        r_k[i][1] = round(nk_n,2)

    global_mean = 0
    for i in range(256):
        global_mean = global_mean+r_k[i][0]*r_k[i][1]
    print("globalmean 1",global_mean)
    global_mean = 0
    for x in range(len(img_arr)):
        for y in range(len(img_arr[0])):
            r = img_arr[x][y]
            p_r = r_k[r][1]
            global_mean = global_mean+r*p_r
    print("globalmean 2",global_mean)
    M_G = np.mean(img_arr)
    print("M_G",M_G)
    global_var = 0
    for x in range(len(img_arr)):
        for y in range(len(img_arr[0])):
            r = img_arr[x][y]
            p_r = r_k[r][1]
            global_var += (r-global_mean)**2*p_r
    global_sd = global_var**0.5
    # mask_size = 3
    step = (mask_size-1)/2
    
    # neighborH = np.zeros((mask_size,mask_size))
    pos_neighbor_33 = np.array([[-1,-1],[-1,0],[-1,1],
                                [0,-1],[0,0],[0,1],
                                [1,-1],[1,0],[1,1]])
    img_new = np.array(image)
    k0,k1,k2 = 0.4,0.02,0.4
    for x in range(1,len(img_arr)-1):
        for y in range(1,len(img_arr[0])-1):
            local_mean = 0 
            r_st = img_arr[x][y]
            #find local mean
            for pos_s_xy in range(len(pos_neighbor_33)):
                s = x+pos_neighbor_33[pos_s_xy][0]
                t = y+pos_neighbor_33[pos_s_xy][1]
                p_r = r_k[img_arr[s][t]][1]
                r = img_arr[s][t]
                local_mean += r*p_r
            #     print("s t",r,p_r)
            # print("localmean",local_mean)
            #find local variance
            local_var = 0
            for pos_s_xy in range(len(pos_neighbor_33)):
                s = x+pos_neighbor_33[pos_s_xy][0]
                t = y+pos_neighbor_33[pos_s_xy][1]
                # if(s<4 and t<4):
                #     print("s t",s,t)
                p_r = r_k[img_arr[s][t]][1]
                r = img_arr[s][t]
                local_var += (r-local_mean)**2*p_r
            # if(x <4 and y <4):
            #     print("===================")
            E = 4
            local_sd = local_var**0.5
            # print(local_mean)
            if(local_mean<=(k0*global_mean) and (k1*global_sd<= local_sd and local_sd <= k2*global_sd)):
                img_new[x][y] = 4*img_arr[x][y]
    return img_new
                    
def localHE2(image,mask_size):
    # compute a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    r_k = np.zeros((256,2))
    img_arr = np.array(image)
    
    for i in range(256):
        r_k[i][0] = i
        
    hist_arr_int = np.array(hist,dtype=int)
    hist_sum = 0
    for i in range(256):
        # print(hist_arr_int[i][0])
        hist_sum += hist_arr_int[i][0]
    #normalize p(r) at r_k[i][1]
    for i in range(256):
        nk_n = hist_arr_int[i][0]/hist_sum
        r_k[i][1] = round(nk_n,2)

    #calculate global_mean
    global_mean = 0
    for i in range(256):
        global_mean = global_mean+r_k[i][0]*r_k[i][1]
    print("globalmean 1",global_mean) #107.82
    
    global_mean = 0
    for x in range(len(img_arr)):
        for y in range(len(img_arr[0])):
            r = img_arr[x][y]
            p_r = r_k[r][1]
            global_mean = global_mean+r*p_r
    print("globalmean 2",global_mean) #204516.95999995034
    
    M_G = np.mean(img_arr)
    global_mean = 0
    for x in range(len(img_arr)):
        for y in range(len(img_arr[0])):
            r = img_arr[x][y]
            p_r = 1/hist_sum
            global_mean = global_mean+r*p_r
    print("globalmean 3",global_mean)
    print("M_G",M_G)
    
    global_var = 0
    for x in range(len(img_arr)):
        for y in range(len(img_arr[0])):
            r = img_arr[x][y]
            p_r = 1/hist_sum
            global_var += (r-global_mean)**2*p_r
    global_sd = global_var**0.5
    D_G = np.std(img_arr)
    print("global sd",global_sd)
    print("D_G",D_G)
    
    
    # mask_size = 3
    step = int((mask_size-1)/2)
    
    # neighborH = np.zeros((mask_size,mask_size))
    pos_neighbor_33 = np.array([[-1,-1],[-1,0],[-1,1],
                                [0,-1],[0,0],[0,1],
                                [1,-1],[1,0],[1,1]])
    img_new = np.array(image)
    k0,k1,k2 = 0.4,0.02,0.4
    for x in range(1,len(img_arr)-1):
        for y in range(1,len(img_arr[0])-1):
            local_mean = 0 
            r_st = img_arr[x][y]
            #find local mean
            for pos_s_xy in range(len(pos_neighbor_33)):
                s = x+pos_neighbor_33[pos_s_xy][0]
                t = y+pos_neighbor_33[pos_s_xy][1]
                p_r = 1/(mask_size**2)
                r = img_arr[s][t]
                local_mean += r*p_r
            #     print("s t",r,p_r)
            # print("localmean",local_mean)
            local = np.zeros((mask_size,mask_size))
            m_sxy = 0
            for i in range(-step,step+1):
                for j in range(-step,step+1):
                    local[1+i][1+j] = img_arr[x+i][y+j]
            m_sxy = np.mean(local)
            std_sxy = np.std(local)
            
            #find local variance
            local_var = 0
            for pos_s_xy in range(len(pos_neighbor_33)):
                s = x+pos_neighbor_33[pos_s_xy][0]
                t = y+pos_neighbor_33[pos_s_xy][1]
                p_r = 1/(mask_size**2)
                r = img_arr[s][t]
                local_var += (r-local_mean)**2*p_r
            
            E = 4
            local_sd = local_var**0.5
            # print("local_mean",round(local_mean,2))
            # print("local_sd",round(local_sd,2))
            # print("m",round(m_sxy,2))
            # print("std sxy",round(std_sxy,2))
            local_mean=round(local_mean,2)
            local_sd=round(local_sd,2)
            m_sxy=round(m_sxy,2)
            std_sxy=round(std_sxy,2)
            # if(round(local_mean,2)==round(m_sxy,2) and round(local_sd,2)==round(std_sxy,2)):
            #     print("true")
            # else:
            #     print("false")
            if(local_mean<=k0*global_mean and (k1*global_sd<= local_sd <= k2*global_sd)):
                img_new[x][y] = min(4*img_arr[x][y],255)
            else:
                img_new[x][y] = img_arr[x][y]
    # Convert the result back to uint8
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    return img_new

def localHE3(image, mask_size,E,k0, k1, k2):
    # Convert image to float32 for more precise calculations
    img_arr = np.array(image, dtype=np.float32)
    path = "/Users/mean/year-4/image-processing/hw-2"
    # Calculate global mean and standard deviation
    global_mean = np.mean(img_arr)
    global_sd = np.std(img_arr)
    print("Global Mean:", global_mean)
    print("Global Standard Deviation:", global_sd)
    np.savetxt(os.path.join(path,f'img_arr.txt'), img_arr, fmt='%d', delimiter=',')
    # Initialize the output image
    img_new = np.array(image, dtype=np.uint8)
    
    # Define the half-step size based on the mask size
    step = int((mask_size - 1) / 2)
    
    # Coefficients
    E, k0, k1, k2= E, k0, k1, k2
    
    #array for Local mean
    arr_localmean = np.zeros((img_arr.shape[0],img_arr.shape[1]))
    arr_localstd = np.zeros((img_arr.shape[0],img_arr.shape[1]))

    # Iterate over each pixel in the image
    for x in range(step, len(img_arr) - step,mask_size-step):
        for y in range(step, len(img_arr[0]) - step,mask_size-step):
            # Extract local window
            
            local_window = img_arr[x-step:x+step+1, y-step:y+step+1]
            # local_arr = img_arr[]
            # Calculate local mean and standard deviation
            local_mean = np.mean(local_window)
            local_sd = np.std(local_window)
            arr_localmean[x][y] = local_mean
            arr_localstd[x][y] = local_sd
            print("Local Mean:", global_mean)
            print("Local Standard Deviation:", global_sd)
            # Condition for modifying pixel intensity
            if local_mean <= k0 * global_mean and k1 * global_sd <= local_sd <= k2 * global_sd:
                img_new[x, y] = min(255, img_arr[x, y] * E)  # Ensure pixel value does not exceed 255
            else:
                img_new[x, y] = img_arr[x, y]
    
    np.savetxt(os.path.join(path,f'arr_localmean.txt'), arr_localmean, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(path,f'arr_localstd.txt'), arr_localstd, fmt='%d', delimiter=',')
    cv2.imwrite(os.path.join(path,f"arr_localmean.jpg"),arr_localmean)
    cv2.imwrite(os.path.join(path,f"arr_localstd.jpg"),arr_localstd)

    # Convert the result back to uint8
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    
    return img_new

def conceptOfLocalHE():
    mask =3
    step =int((mask-1)/2)
    w = 426
    h = 499
    check_outside = []
    for x in range(step,h,mask):
        check = ""
        sub_img = []
        for y in range(step,w,mask):
            walk = y
            check += str(walk)+" "
            # img_arr[x-step:x+step+1, y-step:y+step+1]
            pos_x = np.arange(x-step,x+step+1,1)
            pos_y = np.arange(y-step,y+step+1,1)
            comb_array = np.array(np.meshgrid(pos_x, pos_y)).T.reshape(-1, 2) 

            if x ==1 and y == 1:
                # print(pos_x)
                # print(pos_y)
                print(comb_array)
                
            if x ==1 and y == 424:
                print("this is sub image of 424")
                # print(pos_x)
                # print(pos_y)
                print(comb_array)
                
        if x==1:
            print(check)
        if x==496:
            print("x=496:"+check)
        check_outside.append(x)
        
    # print("this is outside :")
    # print(check_outside)

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
    
    
 
    # # Plotting a basic histogram
    # plt.hist(s_k, bins=256, color='skyblue', edgecolor='black')
    
    # # Adding labels and title
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram')
 
    # # Display the plot
    # plt.show()
    
    #create numpy array of image_original
    img_arr = np.array(image, dtype=np.uint8)
    print(img_arr)
    global_mean = np.mean(img_arr)
    global_std = np.std(img_arr)
    mask =3
    step =int((mask-1)/2)
    h,w = image.shape[0],image.shape[1]
    # w = 426
    # h = 499
    check_outside = []
    
    new_image = np.array(image, dtype=np.uint8)
    
    for x in range(step,h,mask):
        check = ""
        for y in range(step,w,mask):
            walk = y
            check += str(walk)+" "
            #sub image S_xy 
            S_xy = img_arr[x-step:x+step+1, y-step:y+step+1]
            local_mean = np.mean(S_xy)
            local_std  = np.std(S_xy)
            if local_mean <= k0 * global_mean and k1 * global_std <= local_std <= k2 * global_std:
                value = img_arr[x][y]
                
                # new_image[x][y] = s_k[value][1]  # Ensure pixel value does not exceed 255
                # new_image[x][y] = E*img_arr[x][y]  # Ensure pixel value does not exceed 255
                new_image[x-step:x+step+1, y-step:y+step+1] = img_arr[x-step:x+step+1, y-step:y+step+1] * 4
            else:
                new_image[x][y] = img_arr[x][y]
            
        # if x==1:
        #     print(check)
        # if x==496:
        #     print("x=496:"+check)
        # check_outside.append(x)
    print("done")
    return new_image
    
    
image_path = 'Filament.jpg' 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# E, k0, k1, k2 =4, 0.3, 0.05, 0.4
# E, k0, k1, k2 =4, 0.3, 0.05, 0.4
E, k0, k1, k2 =4, 0.45, 0.045, 0.4
# new_img = globalHE(image)
# new_img_LHE = localHE(image,3)
# new_img_LHE = localHE2(image,3)
# new_img_LHE = localHE3(image,3,E, k0, k1, k2)
new_img_LHE = localHistogramEqualization(image,3,E, k0, k1, k2)
# path = "/Users/mean/year-4/image-processing/hw-2"
# cv2.imwrite(os.path.join(path,f"GHE.jpg"),new_img)
path = "/Users/mean/year-4/image-processing/hw-2/test_local_HE"
cv2.imwrite(os.path.join(path,f"LHE-w-E{E}-k0:{k0}-k1:{k1}-k2:{k2}.jpg"),new_img_LHE)
# new_image = im.fromarray(img_new)

# plt.axis('off')  # It helps when Turn off axes to remove the axis ticks and labels
# plt.show(cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB))
# conceptOfLocalHE()
# print(image.shape)