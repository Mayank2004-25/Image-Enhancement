import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Choose enhancement method
print("SELECT THE ENHANCEMENT TECHNIQUES:")
print("1. Fuzzy Set")
print("2. Histogram Equalization")
print("3. Comparison")
choice = int(input("Enter Choice -> "))
image_path = "img11.png"  # Path to your image file

# Metric Calculation for Quality Assessment
def calculate_metrics(original, enhanced):
    psnr_value = cv.PSNR(original, enhanced)
    ssim_value, _ = ssim(original, enhanced, full=True)
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

# Fuzzy Set Enhancement
def fuzzy_set_enhancement(image):
    start_time = time.time()
    n, m = 2, 2
    EPSILON = 0.00001
    GAMMA = 1
    IDEAL_VARIANCE = 0.35
    layer = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    HEIGHT, WIDTH = layer.shape

    def phy(value):
        return 0.5 * np.log((1 + value) / ((1 - value) + EPSILON))

    def scalar_multiplication(scalar, value):
        s = (1 + value) ** scalar
        z = (1 - value) ** scalar
        return (s - z) / (s + z + EPSILON)

    def addition(value1, value2):
        return (value1 + value2) / (1 + value1 * value2 + EPSILON)

    def subtract(value1, value2):
        return (value1 - value2) / (1 - value1 * value2 + EPSILON)

    def mapping(img, source, dest):
        return (dest[1] - dest[0]) * ((img - source[0]) / (source[1] - source[0])) + dest[0]

    e_layer_gray = mapping(layer, (0, 255), (-1, 1))

    def cal_ps_ws(m, n, w, h, gamma):
        ps = np.zeros((m, n, w, h))
        for i in range(m):
            for j in range(n):
                for k in range(w):
                    for l in range(h):
                        ps[i, j, k, l] = 1  # Placeholder
        ws = np.power(ps, gamma) / (np.sum(ps, axis=(0, 1)) + EPSILON)
        return ws

    def one_layer_enhancement(e_layer):
        ws = cal_ps_ws(m, n, WIDTH, HEIGHT, GAMMA)
        new_E_image = e_layer
        res_image = mapping(new_E_image, (-1, 1), (0, 255))
        return res_image.astype(np.uint8)

    res_img = one_layer_enhancement(e_layer_gray)
    
    # Display original and enhanced images with histograms
    plt.subplot(2, 2, 1), plt.imshow(layer, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2, 2, 2), plt.hist(layer.ravel(), 256, [0, 256])
    plt.title('Original Histogram')
    
    plt.subplot(2, 2, 3), plt.imshow(res_img, cmap='gray')
    plt.title('Enhanced Image (Fuzzy Set)')
    plt.subplot(2, 2, 4), plt.hist(res_img.ravel(), 256, [0, 256])
    plt.title('Enhanced Histogram')
    plt.show()
    
    # Metrics Calculation
    calculate_metrics(layer, res_img)
    end_time = time.time()
    print(f"Fuzzy Set Enhancement Time: {end_time - start_time:.2f} seconds")

# Histogram Equalization
def histogram_equalization(image_path):
    start_time = time.time()
    img = Image.open(image_path).convert('L')
    img_array = np.asarray(img)
    histogram, _ = np.histogram(img_array, bins=256, range=(0, 256), density=True)
    cdf = histogram.cumsum()
    transform_map = np.floor(255 * cdf).astype(np.uint8)
    eq_img_array = transform_map[img_array.flatten()].reshape(img_array.shape)
    eq_img = Image.fromarray(eq_img_array)
    
    # Display images and histograms
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.hist(img_array.ravel(), 256, [0, 256])
    plt.title('Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.title('Equalized Image')
    plt.imshow(eq_img, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.hist(eq_img_array.ravel(), 256, [0, 256])
    plt.title('Equalized Histogram')
    plt.show()

    # Calculate Metrics
    calculate_metrics(img_array, eq_img_array)
    end_time = time.time()
    print(f"Histogram Equalization Time: {end_time - start_time:.2f} seconds")

# Comparison
def comparison(image_path):
    image = cv.imread(image_path)
    print("Comparing Fuzzy Set Enhancement and Histogram Equalization")
    fuzzy_set_enhancement(image)
    histogram_equalization(image_path)

# Run the chosen enhancement technique
img = cv.imread(image_path)
if choice == 1:
    fuzzy_set_enhancement(img)
elif choice == 2:
    histogram_equalization(image_path)
elif choice == 3:
    comparison(image_path)
else:
    print("Invalid choice!")
