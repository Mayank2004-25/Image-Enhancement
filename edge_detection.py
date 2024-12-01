import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('img7.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded properly
if image is None:
    raise ValueError("Image not found or unable to open the image file.")

# Convert the image to float32 for better precision in the filtering process
image_float = np.float32(image)

# Sobel edge detection
sobel_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Canny edge detection
canny_edges = cv2.Canny(image, 100, 200)

# Prewitt edge detection (approximated using Sobel kernels)
prewitt_kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewitt_x = cv2.filter2D(image_float, -1, prewitt_kx)
prewitt_y = cv2.filter2D(image_float, -1, prewitt_ky)

# Ensure the Prewitt edge matrices have the correct type (float32) for magnitude calculation
prewitt_x = np.float32(prewitt_x)
prewitt_y = np.float32(prewitt_y)
prewitt_edges = cv2.magnitude(prewitt_x, prewitt_y)

# Roberts edge detection
roberts_kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(image_float, -1, roberts_kx)
roberts_y = cv2.filter2D(image_float, -1, roberts_ky)
roberts_edges = cv2.magnitude(roberts_x, roberts_y)

# Plotting the edge detection results
plt.figure(figsize=(12, 10))

# Sobel
plt.subplot(2, 2, 1)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edge Detection', fontsize=14, pad=20)  # Increased font size and padding
plt.axis('off')

# Canny
plt.subplot(2, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection', fontsize=14, pad=20)
plt.axis('off')

# Prewitt
plt.subplot(2, 2, 3)
plt.imshow(prewitt_edges, cmap='gray')
plt.title('Prewitt Edge Detection', fontsize=14, pad=20)
plt.axis('off')

# Roberts
plt.subplot(2, 2, 4)
plt.imshow(roberts_edges, cmap='gray')
plt.title('Roberts Edge Detection', fontsize=14, pad=20)
plt.axis('off')

plt.tight_layout()
plt.show()
