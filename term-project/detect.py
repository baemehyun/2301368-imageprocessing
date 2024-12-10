import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
# image = cv2.imread('input/Image-1.jpeg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('input/Image-2.jpeg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('input/Image-1.jpeg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('input/Image-4.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the new height
new_height = 500  # Set your desired height

# Calculate the new width to maintain the aspect ratio
(h, w) = image.shape[:2]
aspect_ratio = w / h
new_width = int(new_height * aspect_ratio)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Define the center region (spine area)
center_x, center_y = new_width // 2, new_height // 2  # Calculate center of the image
crop_width, crop_height = new_width // 2, 500  # Width and height of the cropped region

# Define the coordinates for the cropping
x_start = max(center_x - crop_width // 2, 0)
x_end = min(center_x + crop_width // 2, new_width)
y_start = max(center_y - crop_height // 2, 0)
y_end = min(center_y + crop_height // 2, new_height)

# # Create a binary mask for the cropped region
# mask = np.zeros_like(resized_image, dtype=np.uint8)
# mask[y_start:y_end, x_start:x_end] = 255
# # Apply the mask to extract the spine area
# spine_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

# Extract the cropped region
cropped_spine = resized_image[y_start:y_end, x_start:x_end]
# Sharpen the image using a kernel
sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
sharpened_image = cv2.filter2D(cropped_spine, -1, sharpening_kernel)


# Calculate gradients using Sobel operators
sobel_x = cv2.Sobel(resized_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
sobel_y = cv2.Sobel(resized_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

# Calculate gradient magnitude
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Cropped spine image
# Calculate gradients using Sobel operators
sobel_x = cv2.Sobel(sharpened_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
sobel_y = cv2.Sobel(sharpened_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction

# Calculate gradient magnitude
gradient_magnitude_cropped_spine = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude_cropped_spine = cv2.convertScaleAbs(gradient_magnitude_cropped_spine)

# Combine the original image with the gradient magnitude
# combined_image = cv2.addWeighted(resized_image, 0.5, gradient_magnitude, 0.5, 0)
combined_image = cv2.addWeighted(resized_image, 0.3, gradient_magnitude, 0.7, 0)
# combined_cropped_spine = cv2.addWeighted(cropped_spine, 0.5, gradient_magnitude_cropped_spine, 0.5, 0)
combined_cropped_spine = cv2.addWeighted(cropped_spine, 0.4, gradient_magnitude_cropped_spine, 0.6, 0)

# Define thresholds
low_threshold = 100
high_threshold = 155

# Apply double thresholding
strong_edges = (combined_cropped_spine >= high_threshold).astype(np.uint8) * 255  # Strong edges
weak_edges = ((combined_cropped_spine >= low_threshold) & (combined_cropped_spine < high_threshold)).astype(np.uint8) * 128  # Weak edges

# Combine strong and weak edges into a single image
double_thresholded_image = strong_edges + weak_edges
# Parameters: low_threshold and high_threshold for edge detection
_, otsu_thresholded = cv2.threshold(combined_cropped_spine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
spine_Canny = cv2.Canny(double_thresholded_image, threshold1=100, threshold2=150)
median_filtered_image = cv2.medianBlur(otsu_thresholded, 3)  # Kernel size 5


# Display results
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 2)
plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 3)
plt.title('Combined Image (Original + Gradient Magnitude)')
plt.imshow(combined_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 4)
plt.title('cropped spine image')
plt.imshow(cropped_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 5)
plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude_cropped_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 6)
plt.title('Combined (cropped + Gradient Magnitude)')
plt.imshow(sharpened_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 7)
plt.title('Double threshold')
plt.imshow(double_thresholded_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 8)
plt.title('spine with canny')
plt.imshow(spine_Canny, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 9)
plt.title('spine with canny')
plt.imshow(median_filtered_image, cmap='gray')
plt.axis('off')
plt.show()

binary_image = np.array(median_filtered_image)


def conditionDialation(image,num=3):
    kernel = np.ones((3, 1), np.uint8)
    erodedCD_image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((num, num), np.uint8)
    dialatedCD_image = cv2.dilate(erodedCD_image, kernel, iterations=1)
    # Perform bitwise AND operation to calculate the intersection
    intersection = cv2.bitwise_and(image, dialatedCD_image)
    return intersection
def remove_small_areas(image, min_area=50):
    # Convert to binary if the image is not already binary
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(image, dtype=np.uint8)

    # Iterate through contours and keep only those with an area >= min_area
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)  # Fill the retained region

    # Apply the mask to the image
    filtered_image = cv2.bitwise_and(image, mask)

    return filtered_image

gaussian_filtered_median = cv2.GaussianBlur(median_filtered_image, (5, 5), 0)
_, otsu_thresholded = cv2.threshold(gaussian_filtered_median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


rsa_median_filtered_image = remove_small_areas(otsu_thresholded,500)
# Apply closing (dilation followed by erosion)
kernel = np.ones((5, 5), np.uint8)  # Define the kernel for morphological operations
closed_image = cv2.morphologyEx(rsa_median_filtered_image, cv2.MORPH_CLOSE, kernel)
intersec1 = conditionDialation(closed_image,7 )
intersec2 = conditionDialation(intersec1,7)
intersec3 = conditionDialation(intersec2,7)
filter_intersec3 = remove_small_areas(intersec3)
filter_rsa = remove_small_areas(filter_intersec3,1000)
cd_filter_rsa = conditionDialation(filter_rsa,5)
filter_closed_image = remove_small_areas(closed_image)
median_filtered_cd_filter_rsa = cv2.medianBlur(cd_filter_rsa,5)
median_intersec3 = cv2.medianBlur(intersec3, 5)  # Kernel size 5
# Define the kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# Apply opening (erosion followed by dilation)
opened_image = cv2.morphologyEx(median_filtered_cd_filter_rsa, cv2.MORPH_OPEN, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
# # Apply opening (erosion followed by dilation)
# opened_image = cv2.morphologyEx(median_filtered_cd_filter_rsa, cv2.MORPH_OPEN, kernel)
filter_opened_image = remove_small_areas(opened_image,100)
cd_filter_opened_image = conditionDialation(filter_opened_image,5)
kernel = np.ones((9, 9), np.uint8)
dilated_cd_filter_opened_image = cv2.dilate(cd_filter_opened_image, kernel, iterations=1)
filter_dilated_cd = cv2.medianBlur(dilated_cd_filter_opened_image, 7)
# Apply closing (dilation followed by erosion)
kernel = np.ones((7, 7), np.uint8)  # Define the kernel for morphological operations
closed_filter_dilated_cd = cv2.morphologyEx(filter_dilated_cd, cv2.MORPH_CLOSE, kernel)
# kernel = np.ones((7, 7), np.uint8)  # Define the kernel for morphological operations
# closed_filter_dilated_cd = cv2.morphologyEx(closed_filter_dilated_cd, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((5, 5), np.uint8)
eroded_cd_filter_opened_image = cv2.erode(closed_filter_dilated_cd,kernel,iterations=1)
# kernel = np.ones((7, 7), np.uint8)
# eroded_cd_filter_opened_image = cv2.erode(eroded_cd_filter_opened_image,kernel,iterations=1)

def calculate_spine_width_row(binary_image):
    widths = []
    for y in range(binary_image.shape[0]):  # Loop through each row
        row = binary_image[y, :]
        if 255 in row:  # Check if there's any white pixel in the row
            left = np.min(np.where(row == 255))  # Leftmost white pixel
            right = np.max(np.where(row == 255))  # Rightmost white pixel
            width = right - left + 1  # Calculate width
        else:
            width = 0  # No spine in this row
        widths.append(width)
    return widths

def reduce_spine_to_median_width(binary_image, spine_widths,edge_margin=10):
    # Calculate the median width
    median_width = int(np.median([w for w in spine_widths if w > 0]))  # Ignore rows with zero width
    max_width = max(spine_widths)

    # Calculate the median width
    median_width = int(np.median([w for w in spine_widths if w > 0]))  # Ignore rows with zero width

    # Create a copy of the binary image to modify
    reduced_spine_image = binary_image.copy()
    max_width = binary_image.shape[1]

    for y in range(binary_image.shape[0]):  # Loop through each row
        row = binary_image[y, :]
        if 255 in row:  # Check if there's any white pixel in the row
            left = np.min(np.where(row == 255))  # Leftmost white pixel
            right = np.max(np.where(row == 255))  # Rightmost white pixel
            current_width = right - left + 1

            # Check if the row is an edge row
            if left < 10 or right > max_width - 10:  # Close to edges
                if current_width > median_width:  # Reduce width to the median
                    excess = current_width - median_width
                    left_trim = excess // 2
                    right_trim = excess - left_trim
                    reduced_spine_image[y, left:left + left_trim] = 0  # Remove excess pixels from the left
                    reduced_spine_image[y, right - right_trim + 1:right + 1] = 0  # Remove excess pixels from the right

    return reduced_spine_image, median_width



find_widths_spine = calculate_spine_width_row(eroded_cd_filter_opened_image)
reduce_spine,median_width = reduce_spine_to_median_width(eroded_cd_filter_opened_image,find_widths_spine)
# Create a morphological kernel based on the median width
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (median_width//2, median_width))

# Apply morphological opening (erosion followed by dilation)
opened_eroded_cd_median = cv2.morphologyEx(eroded_cd_filter_opened_image, cv2.MORPH_OPEN, kernel)
opened_eroded_cd_median = remove_small_areas(opened_eroded_cd_median,2000)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel = np.ones((9, 9), np.uint8)  # Define the kernel for morphological operations
dilate_opened_eroded_cd_median = cv2.dilate(opened_eroded_cd_median, kernel, iterations=1)
# kernel = np.ones((5, 5), np.uint8)  # Define the kernel for morphological operations
# dilate_opened_eroded_cd_median = cv2.dilate(dilate_opened_eroded_cd_median, kernel, iterations=1)
# dilate_opened_eroded_cd_median = cv2.morphologyEx(eroded_cd_filter_opened_image, cv2.MORPH_OPEN, kernel)

spine_mask=cv2.bitwise_and(dilate_opened_eroded_cd_median,eroded_cd_filter_opened_image)

reduce_spine = remove_small_areas(reduce_spine,500)
kernel = np.ones((15, 15), np.uint8)  # Define the kernel for morphological operations
closed_filter_reduce_spine = cv2.morphologyEx(reduce_spine, cv2.MORPH_CLOSE, kernel)
guassian_closed_reduce_spine = cv2.GaussianBlur(closed_filter_reduce_spine,(5,5),0)

pre_spine_mask = cv2.bitwise_or(guassian_closed_reduce_spine,opened_eroded_cd_median)
# Create a morphological kernel based on the median width
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (median_width//2, median_width))
pre_spine_mask = cv2.morphologyEx(pre_spine_mask, cv2.MORPH_OPEN, kernel)
pre_spine_mask = remove_small_areas(pre_spine_mask,2000)
fusion_premask_prepairSpine = cv2.bitwise_or(eroded_cd_filter_opened_image,pre_spine_mask)
# kernel = np.ones((7, 7), np.uint8)
# pre_spine_mask = cv2.dilate(pre_spine_mask, kernel, iterations=1)




spine_mask = cv2.bitwise_and(fusion_premask_prepairSpine,pre_spine_mask)

def connect_components(binary_image):
    # Find connected components and their centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    if num_labels > 2:  # More than 2 components (background + two objects)
        # Get the centroids of the first two components (ignoring background)
        print(centroids)
        centroid1 = tuple(map(int, centroids[2]+25))
        centroid2 = tuple(map(int, centroids[1]))

        # Draw a line connecting the two centroids
        connected_image = binary_image.copy()
        cv2.line(connected_image, centroid1, centroid2, 255, thickness=1)  # Adjust thickness as needed
        
        return connected_image

    # If less than two components, return the original image
    return binary_image
def calculate_spine_width_row(binary_image):

    widths = []
    centers = []
    for y in range(binary_image.shape[0]):  # Loop through each row
        row = binary_image[y, :]
        if 255 in row:  # Check if there's any white pixel in the row
            left = np.min(np.where(row == 255))  # Leftmost white pixel
            right = np.max(np.where(row == 255))  # Rightmost white pixel
            width = right - left + 1  # Calculate width
            center = (left + right) // 2  # Calculate center
        else:
            width = 0  # No spine in this row
            center = None
        widths.append(width)
        centers.append(center)
    return widths, centers

def adjust_spine_width(binary_image, widths, centers, mean_width):
    adjusted_spine = binary_image.copy()

    for y in range(0, binary_image.shape[0]):  # Start from the 4th row to reference the above rows
        if centers[y] is not None:
            # Calculate the mean center of the above 4 rows
            above_centers = [c for c in centers[y-1:y] if c is not None]
            if above_centers:
                mean_center = int(np.mean(above_centers))
            else:
                mean_center = centers[y]

            # Adjust the row width
            if widths[y] < mean_width:  # Expand row
                start = max(0, mean_center - mean_width // 2)
                end = min(binary_image.shape[1], mean_center + mean_width // 2)
                adjusted_spine[y, :] = 0  # Clear the row
                adjusted_spine[y, start:end + 1] = 255  # Expand the spine
            elif widths[y] > mean_width:  # Reduce row
                excess = widths[y] - mean_width
                start = max(0, centers[y] - mean_width // 2)
                end = min(binary_image.shape[1], centers[y] + mean_width // 2)
                adjusted_spine[y, :] = 0  # Clear the row
                adjusted_spine[y, start:end + 1] = 255  # Reduce the spine

    return adjusted_spine

check_spine_mask = connect_components(spine_mask)

find_widths_spine_mask,find_center_spine_mask = calculate_spine_width_row(check_spine_mask)
# Calculate the mean width of the spine
mean_width = int(np.mean([w for w in find_widths_spine_mask if w > 0]))  # Ignore rows with zero width
check_spine_mask = adjust_spine_width(check_spine_mask,find_widths_spine_mask,find_center_spine_mask,mean_width)
kernel = np.ones((3, 3), np.uint8)
spine_mask = cv2.dilate(check_spine_mask,kernel, iterations=1)

kernel = np.ones((3, 3), np.uint8)
dialated_median_intersec3 = cv2.dilate(median_intersec3, kernel, iterations=1)

combined_spine_filtered = cv2.addWeighted(cropped_spine, 0.5, closed_filter_dilated_cd, 0.5, 0)

intersec4 = cv2.bitwise_and(cropped_spine, closed_filter_dilated_cd)
intersec4 = cv2.bitwise_and(cropped_spine, guassian_closed_reduce_spine)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))  # Define CLAHE parameters
clahe_image = clahe.apply(combined_spine_filtered)

################# prepair-mask
# Display results
plt.figure(figsize=(15, 10))
plt.subplot(3, 5, 1)
plt.title('start')
plt.imshow(median_filtered_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 2)
plt.title('closing')
plt.imshow(closed_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 3)
plt.title('CD 3')
plt.imshow(intersec3, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 4)
plt.title('filter_intersec3')
plt.imshow(filter_intersec3, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 5)
plt.title('filter_closed_image')
plt.imshow(filter_closed_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 6)
plt.title('cd_filter_rsa')
plt.imshow(cd_filter_rsa, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 7)
plt.title('median_filtered_filter_closed_image')
plt.imshow(median_filtered_cd_filter_rsa, cmap='gray')
plt.axis('off')
# plt.subplot(3, 5, 8)
# plt.title('median_filtered_image')
# plt.imshow(median_filtered_image, cmap='gray')
# plt.axis('off')
plt.subplot(3, 5, 8)
plt.title('opened_image ')
plt.imshow(opened_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 9)
plt.title('dilated_cd_filter_opened_image')
plt.imshow(dilated_cd_filter_opened_image, cmap='gray')
plt.axis('off')
plt.subplot(3,5, 10)
plt.title('eroded_cd_filter_opened_image')
plt.imshow(eroded_cd_filter_opened_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 11)
plt.title(f'reduce_spine {median_width}')
plt.imshow(reduce_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 12)
plt.title(f'guassian_closed_reduce_spine')
plt.imshow(guassian_closed_reduce_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 15)
plt.title('spine_mask')
plt.imshow(spine_mask, cmap='gray')
plt.axis('off')
plt.axis('off')
plt.subplot(3,5, 14)
plt.title('dilate_opened_eroded_cd_median')
plt.imshow(dilate_opened_eroded_cd_median, cmap='gray')
plt.axis('off')
plt.subplot(3,5, 13)
plt.title('opened_eroded_cd_median')
plt.imshow(opened_eroded_cd_median, cmap='gray')
plt.axis('off')
plt.show()

################# prepair-mask

# # Apply Gaussian filtering to the combined image
# gaussian_filtered_image = cv2.GaussianBlur(combined_image, (5, 5), 0)

# # Apply Otsu's method for thresholding
# _, otsu_thresholded = cv2.threshold(gaussian_filtered_image, 0, 255, cv2.THRESH_OTSU)

# # Display results
# plt.figure(figsize=(15, 10))

# # Gradient images
# plt.subplot(2, 3, 1)
# plt.title('Gradient X (Sobel X)')
# plt.imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.title('Gradient Y (Sobel Y)')
# plt.imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.title('Gradient Magnitude')
# plt.imshow(gradient_magnitude, cmap='gray')
# plt.axis('off')

# # Original and processed images
# plt.subplot(2, 3, 4)
# plt.title('Original Image')
# plt.imshow(resized_image, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 5)
# plt.title('Gaussian Filtered Image')
# plt.imshow(gaussian_filtered_image, cmap='gray')
# plt.axis('off')

# # Otsu's thresholding result
# plt.subplot(2, 3, 6)
# plt.title("Otsu's Thresholding")
# plt.imshow(otsu_thresholded, cmap='gray')
# plt.axis('off')

# plt.show()


# Apply Laplacian of Gaussian (LoG) filter
# First, apply Gaussian smoothing
gaussian_blurred = cv2.GaussianBlur(combined_image, (3, 3), 0)

# Then, apply Laplacian filter
log_filtered_image = cv2.Laplacian(gaussian_blurred, cv2.CV_64F, ksize=3)
log_filtered_image = cv2.convertScaleAbs(log_filtered_image)


gaussian_blurred_2 = cv2.GaussianBlur(log_filtered_image, (3, 3), 0)

# Apply Otsu's method for thresholding on LoG-filtered image
_, otsu_thresholded = cv2.threshold(gaussian_blurred_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Canny edge detection
# Parameters: low_threshold and high_threshold for edge detection
edgesCanny = cv2.Canny(gaussian_blurred, threshold1=80, threshold2=150)

# Apply Adaptive Gaussian Thresholding on LoG-filtered image
adaptive_thresholded = cv2.adaptiveThreshold(
    combined_image, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    3, 7  # Block size and constant C
)


# # Apply mean filtering to the Otsu-thresholded image
# mean_filtered_image = cv2.blur(otsu_thresholded, (3, 3))
# Apply median filtering to the Otsu-thresholded image
median_filtered_image = cv2.medianBlur(otsu_thresholded, 3)  # Kernel size 5

# Apply dilation to the Otsu-thresholded image
kernel = np.ones((3, 3), np.uint8)  # Define the kernel for dilation

dilated_image = cv2.dilate(median_filtered_image, kernel, iterations=1)

# Apply erosion to the median-filtered image
kernel = np.ones((3, 3), np.uint8)  # Define the kernel for morphological operations
eroded_image = cv2.erode(median_filtered_image, kernel, iterations=1)

# Display results
# plt.figure(figsize=(15, 10))

# # Original and processed images
# plt.subplot(3, 3, 1)
# plt.title('Original Image')
# plt.imshow(resized_image, cmap='gray')
# plt.axis('off')

# plt.subplot(3, 3, 2)
# plt.title('Combined Image (Original + Gradient Magnitude)')
# plt.imshow(combined_image, cmap='gray')
# plt.axis('off')

# plt.subplot(3, 3, 3)
# plt.title('Gaussian Blurred Image')
# plt.imshow(gaussian_blurred, cmap='gray')
# plt.axis('off')

# plt.subplot(3, 3, 4)
# plt.title('LoG Filtered Image')
# plt.imshow(log_filtered_image, cmap='gray')
# plt.axis('off')

# # Otsu's Thresholding result
# plt.subplot(3, 3, 5)
# plt.title("Otsu's Thresholding")
# plt.imshow(otsu_thresholded, cmap='gray')
# plt.axis('off')

# # # Adaptive Gaussian Thresholding result
# # plt.subplot(3, 3, 6)
# # plt.title("Adaptive Gaussian Thresholding")
# # plt.imshow(adaptive_thresholded, cmap='gray')
# # plt.axis('off')

# plt.subplot(3, 3, 6)
# plt.title('Canny Edge Detection')
# plt.imshow(edgesCanny, cmap='gray')
# plt.axis('off')


# # Median filtering result
# plt.subplot(3, 3, 7)
# plt.title("Median Filtered Image")
# plt.imshow(median_filtered_image, cmap='gray')
# plt.axis('off')

# # Dilated image
# plt.subplot(3, 3, 8)
# plt.title("Dilated Image")
# plt.imshow(dilated_image, cmap='gray')
# plt.axis('off')

# plt.show()

####### after get spine mask


kernel = np.ones((5, 5), np.uint8)
spine_mask = cv2.dilate(check_spine_mask,kernel, iterations=1)
unblue_spine_mask = spine_mask.copy()
spine_mask = cv2.GaussianBlur(spine_mask, (15, 15), 0)


intersect_spine_mask = cv2.bitwise_and(spine_mask,combined_cropped_spine)

intersect_spine_mask = cv2.GaussianBlur(intersect_spine_mask, (3, 3), 0)

combined_spine_mask = cv2.addWeighted(cropped_spine, 0.5, intersect_spine_mask, 0.4, 0)
# Apply Otsu's method for thresholding on LoG-filtered image
_, otsu_combined_spine_mask = cv2.threshold(combined_spine_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))  # Define CLAHE parameters
clahe_sharpened_image = clahe.apply(combined_cropped_spine)
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(15, 15))  # Define CLAHE parameters
clahe_combined_spine = clahe.apply(combined_spine_mask)
intersec_clahe_combined_spine = cv2.bitwise_and(clahe_combined_spine,unblue_spine_mask)
preprocess_spine = cv2.addWeighted(intersec_clahe_combined_spine, 0.3, cropped_spine, 0.5, 0)
preprocess_spine = cv2.bitwise_and(preprocess_spine, unblue_spine_mask)

intersect_cropped_spine = cv2.bitwise_and(unblue_spine_mask,cropped_spine)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(5, 5))
intersect_cropped_spine = clahe.apply(intersect_cropped_spine)
# _, intersect_cropped_spine = cv2.threshold(intersect_cropped_spine, 140, 255, cv2.THRESH_BINARY)
# intersect_cropped_spine = cv2.GaussianBlur(intersect_cropped_spine,(7,7),0)
# preprocess_spine = cv2.addWeighted(intersec_clahe_sharp, 0.5, gradient_magnitude_spine, 0.5, 0)
# sharpening_kernel = np.array([[0, -1, 0],
#                                [-1, 5, -1],
#                                [0, -1, 0]])
# preprocess_spine = cv2.filter2D(preprocess_spine, -1, sharpening_kernel)

filter_preprocess_spine = cv2.GaussianBlur(intersec_clahe_combined_spine,(7,7),0)
_, otsu_preprocess_spine = cv2.threshold(intersec_clahe_combined_spine, 110, 255, cv2.THRESH_BINARY)
otsu_preprocess_spine = cv2.GaussianBlur(otsu_preprocess_spine,(7,7),0)

# preprocess_spine_Canny = cv2.Canny(preprocess_spine, threshold1=100, threshold2=150)

# Display results
plt.figure(figsize=(15, 10))
plt.subplot(3, 5, 1)
plt.title('start spine_mask')
plt.imshow(spine_mask, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 2)
plt.title('intersect_cropped_spine')
plt.imshow(intersect_cropped_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 3)
plt.title('intersect_spine_mask')
plt.imshow(intersect_spine_mask, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 4)
plt.title('combined_spine_mask')
plt.imshow(combined_spine_mask, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 5)
plt.title('otsu_combined_spine_mask')
plt.imshow(otsu_combined_spine_mask, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 6)
plt.title('clahe_sharpened_image')
plt.imshow(clahe_sharpened_image, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 7)
plt.title('clahe_combined_spine')
plt.imshow(clahe_combined_spine, cmap='gray')
plt.axis('off')
# plt.subplot(3, 5, 8)
# plt.title('median_filtered_image')
# plt.imshow(median_filtered_image, cmap='gray')
# plt.axis('off')
plt.subplot(3, 5, 8)
plt.title('intersec_clahe_combined_spine ')
plt.imshow(intersec_clahe_combined_spine, cmap='gray')
plt.axis('off')
plt.subplot(3, 5, 9)
plt.title('preprocess_spine')
plt.imshow(preprocess_spine, cmap='gray')
plt.axis('off')
plt.subplot(3,5, 10)
plt.title('otsu_preprocess_spine')
plt.imshow(otsu_preprocess_spine, cmap='gray')


blur_clahe_sharpened_image = cv2.GaussianBlur(clahe_combined_spine,(5, 5), 0)
# Calculate gradients using Sobel operators
sobel_x = cv2.Sobel(blur_clahe_sharpened_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
sobel_y = cv2.Sobel(blur_clahe_sharpened_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
# Calculate gradient magnitude
gradient_magnitude_spine = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude_spine = cv2.convertScaleAbs(gradient_magnitude_spine)
gradient_magnitude_spine = cv2.bitwise_and(gradient_magnitude_spine,spine_mask)

plt.axis('off')
plt.subplot(3, 5, 11)
plt.title(f'gradient_magnitude_spine ')
plt.imshow(gradient_magnitude_spine, cmap='gray')
plt.axis('off')

# intersec_clahe_combined_spine
combine_intersecClahe_with_cropped = cv2.addWeighted(intersec_clahe_combined_spine,0.4,cropped_spine,0.4,0)

plt.subplot(3, 5, 12)
plt.title(f'combine_intersecClahe_with_cropped')
plt.imshow(combine_intersecClahe_with_cropped, cmap='gray')
plt.axis('off')

def get_spine_center_line(binary_spine):

    # Create a blank image to draw the spine center line
    center_line_image = np.zeros_like(binary_spine)

    # Get the center points of the spine for each row
    center_points = []
    for y in range(binary_spine.shape[0]):
        row = binary_spine[y, :]
        if 255 in row:  # Check if there are spine pixels in the row
            left = np.min(np.where(row == 255))  # Leftmost white pixel
            right = np.max(np.where(row == 255))  # Rightmost white pixel
            center = (left + right) // 2  # Calculate the center of the row
            center_points.append((center, y))

    # Draw the center line on the blank image
    for i in range(1, len(center_points)):
        cv2.line(center_line_image, center_points[i - 1], center_points[i], 255, thickness=2)

    return center_line_image, center_points
# Generate the spine center line from the binary spine mask
spine_center_line, spine_center_points = get_spine_center_line(unblue_spine_mask)

def draw_bounding_boxes_following_spine(binary_spine, center_points, box_width, box_height, row_gap):
    # Create a blank image to draw the bounding boxes
    bounding_box_image = cv2.cvtColor(binary_spine, cv2.COLOR_GRAY2BGR)

    # Track the last drawn bounding box row to enforce the gap
    last_y = -row_gap
    end_y = 0
    for x, y in center_points:
        # Ensure the current box is at least 'row_gap' rows away from the last one
        
        if y - last_y >= row_gap and y >= end_y:
            # Calculate the box coordinates
            x_start = max(0, x - box_width // 2)
            x_end = min(binary_spine.shape[1], x + box_width // 2)
            y_start = max(0, y - box_height // 2)
            y_end = min(binary_spine.shape[0], y + box_height // 2)
            end_y = y_end
            print((x_start, y_start), (x_end, y_end))
            # Draw the bounding box
            # cv2.rectangle(bounding_box_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
            cv2.line(bounding_box_image,(x - box_width // 2,y),(x + box_width // 2,y),(255, 0, 0),2)
            # Update the last drawn box row
            last_y = y

    return bounding_box_image

# Parameters for bounding box dimensions and spacing
box_width = mean_width+15  # Width of the bounding box
box_height = 500//20  # Height of the bounding box
row_gap = 16  # Minimum row gap between consecutive boxes

# Draw bounding boxes following the spine center points
bounding_box_image = draw_bounding_boxes_following_spine(intersect_cropped_spine, spine_center_points, box_width, box_height, row_gap)


def sobel_edge_detection(image):
    """
    Applies Sobel edge detection to an input grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Image showing the detected edges.
    """
    # Apply Sobel filter in the X direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)  # Convert back to 8-bit

    # Apply Sobel filter in the Y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)  # Convert back to 8-bit

    # Combine the two gradients
    sobel_combined = cv2.addWeighted(sobel_x, 0.6, sobel_y, 0.3, 0)

    return sobel_combined

result_sobel = sobel_edge_detection(intersec_clahe_combined_spine)
result_sobel = sobel_edge_detection(preprocess_spine)
result_sobel = sobel_edge_detection(intersect_cropped_spine)

plt.subplot(3, 5, 13)
plt.title('result_sobel')
plt.imshow(result_sobel, cmap='gray')
plt.axis('off')

def prewitt_edge_detection(image):

    # Apply Prewitt filter in the X direction
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_x = cv2.filter2D(image, -1, kernel_x)

    # Apply Prewitt filter in the Y direction
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Combine the two gradients
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    return prewitt_combined

# Apply Prewitt edge detection
prewitt_edges = prewitt_edge_detection(intersect_cropped_spine)

plt.subplot(3,5, 14)
plt.title('bounding_box_image')
plt.imshow(bounding_box_image, cmap='gray')
plt.axis('off')
def detect_squares(binary_image):

    # Convert the binary image to BGR for visualization
    square_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected square contours
    squares = []

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices and is convex (a potential square or rectangle)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate aspect ratio to check if it approximates a square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:  # Aspect ratio close to 1
                squares.append(approx)
                # Draw the square on the image
                cv2.drawContours(square_image, [approx], -1, (0, 255, 0), 2)

    return square_image, squares

# Apply the square detection function
square_detected_image, detected_squares = detect_squares(intersect_cropped_spine)

plt.subplot(3,5, 15)
plt.title('square_detected_image')
plt.imshow(square_detected_image, cmap='gray')
plt.axis('off')
plt.show()


