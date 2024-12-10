from skimage import io, color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the image
image = io.imread('input/Image-1.jpeg')
gray_image = color.rgb2gray(image)

# Split the image into top and bottom halves
height = gray_image.shape[0]
top_half = gray_image[:height//2, :]
bottom_half = gray_image[height//2:, :]

# Apply filters locally
def apply_filters(image):
    """Apply different filters to the image."""
    filters = [
        (original, None),
        (meijering, [1]),
        (meijering, range(1, 5)),
        (sato, [1]),
        (sato, range(1, 5)),
        (frangi, [1]),
        (frangi, range(1, 5)),
        (hessian, [1]),
        (hessian, range(1, 5)),
    ]
    results = []
    for func, sigmas in filters:
        result = func(image, black_ridges=True, sigmas=sigmas)
        results.append((func.__name__, sigmas, result))
    return results

# Function to return original image
def original(image, **kwargs):
    return image

# Process top and bottom halves
top_results = apply_filters(top_half)
bottom_results = apply_filters(bottom_half)

# Combine results
cmap = plt.cm.gray
plt.rcParams["axes.titlesize"] = "medium"

# Display results
fig, axes = plt.subplots(2, 9, figsize=(12, 5))
for i, results in enumerate([top_results, bottom_results]):
    for j, (name, sigmas, result) in enumerate(results):
        axes[i, j].imshow(result, cmap=cmap)
        title = f"{name}"
        if sigmas:
            title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
        axes[i, j].set_title(title)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
