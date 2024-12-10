from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'input/Image-1.jpeg'
image = Image.open(image_path).convert("L")  # Convert to grayscale

# Convert image to numpy array
image_array = np.array(image)

# Split the image into top and bottom halves
height = image_array.shape[0]
mid_point = height // 2
top_half = image_array[:mid_point, :]
bottom_half = image_array[mid_point:, :]

# Define a function for histogram equalization
def histogram_equalization(image_section):
    hist, bins = np.histogram(image_section.flatten(), bins=256, range=[0,255])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf[image_section]
    return equalized_image, hist, cdf

# Apply histogram equalization to top and bottom halves
equalized_top, hist_top, cdf_top = histogram_equalization(top_half)
equalized_bottom, hist_bottom, cdf_bottom = histogram_equalization(bottom_half)

# Combine the equalized top and bottom halves
equalized_image = np.vstack((equalized_top, equalized_bottom))

# Plot the original and equalized images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Top Half")
plt.imshow(top_half, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Histogram of Original Top")
plt.hist(top_half.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 3)
plt.title("CDF of Original Top")
plt.plot(cdf_top, color='blue')
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")

plt.subplot(2, 3, 4)
plt.title("Equalized Top Half")
plt.imshow(equalized_top, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Histogram of Equalized Top")
plt.hist(equalized_top.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 6)
plt.title("CDF of Equalized Top")
plt.hist(equalized_top.flatten(), bins=256, range=[0, 256], cumulative=True, color='blue', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Bottom Half")
plt.imshow(bottom_half, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Histogram of Original Bottom")
plt.hist(bottom_half.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 3)
plt.title("CDF of Original Bottom")
plt.plot(cdf_bottom, color='blue')
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")

plt.subplot(2, 3, 4)
plt.title("Equalized Bottom Half")
plt.imshow(equalized_bottom, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Histogram of Equalized Bottom")
plt.hist(equalized_bottom.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 6)
plt.title("CDF of Equalized Bottom")
plt.hist(equalized_bottom.flatten(), bins=256, range=[0, 256], cumulative=True, color='blue', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")

plt.tight_layout()
plt.show()
