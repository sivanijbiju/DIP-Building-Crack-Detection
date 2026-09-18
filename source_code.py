import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. Load Original Crack Images
# ============================================================

image_names = [
    "crack_01.jpg",
    "crack_02.jpg",
    "crack_03.jpg",
    "crack_04.jpg",
    "crack_05.jpg",
    "crack_06.jpg",
    "crack_07.jpg"
]

image_folder = "images"

images = []

for name in image_names:
    image = cv2.imread(f"{image_folder}/{name}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)


plt.figure(figsize=(12, 8))

for i, image in enumerate(images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.title("Original " + str(i + 1))
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 2. Convert Images to Grayscale
# ============================================================

gray_images = []

for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_images.append(gray)


plt.figure(figsize=(12, 8))

for i, gray in enumerate(gray_images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale " + str(i + 1))
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 3. Add Salt-and-Pepper Noise
# ============================================================

def add_salt_pepper_noise(image, amount=0.05):

    noisy = image.copy()

    total_pixels = image.size

    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    coords = [
        np.random.randint(0, i - 1, num_salt)
        for i in image.shape
    ]

    noisy[tuple(coords)] = 255

    coords = [
        np.random.randint(0, i - 1, num_pepper)
        for i in image.shape
    ]

    noisy[tuple(coords)] = 0

    return noisy


noisy_images = []

for gray in gray_images:

    noisy = add_salt_pepper_noise(gray, 0.05)

    noisy_images.append(noisy)


plt.figure(figsize=(12, 8))

for i, noisy in enumerate(noisy_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(noisy, cmap="gray")

    plt.title("Noisy Image " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 4. Mean Filter
# ============================================================

mean_images = []

for noisy in noisy_images:

    mean = cv2.blur(noisy, (5, 5))

    mean_images.append(mean)


plt.figure(figsize=(12, 8))

for i, mean in enumerate(mean_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(mean, cmap="gray")

    plt.title("Mean Filter " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 5. Median Filter
# ============================================================

median_images = []

for noisy in noisy_images:

    median = cv2.medianBlur(noisy, 5)

    median_images.append(median)


plt.figure(figsize=(12, 8))

for i, median in enumerate(median_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(median, cmap="gray")

    plt.title("Median Filter " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 6. Gaussian Filter
# ============================================================

gaussian_images = []

for noisy in noisy_images:

    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)

    gaussian_images.append(gaussian)


plt.figure(figsize=(12, 8))

for i, gaussian in enumerate(gaussian_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(gaussian, cmap="gray")

    plt.title("Gaussian Filter " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 7. Restoration Comparison
# ============================================================

image_index = 0

plt.figure(figsize=(12, 4))


plt.subplot(1, 4, 1)

plt.imshow(noisy_images[image_index], cmap="gray")

plt.title("Noisy Image")

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(mean_images[image_index], cmap="gray")

plt.title("Mean Filter")

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(median_images[image_index], cmap="gray")

plt.title("Median Filter")

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(gaussian_images[image_index], cmap="gray")

plt.title("Gaussian Filter")

plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 8. Calculate MSE, PSNR and SSIM
# ============================================================

mse_mean = []
mse_median = []
mse_gaussian = []

psnr_mean = []
psnr_median = []
psnr_gaussian = []

ssim_mean = []
ssim_median = []
ssim_gaussian = []


for i in range(len(gray_images)):

    original = gray_images[i]


    # Mean Filter Metrics

    mse_m = np.mean(
        (original.astype(float) -
         mean_images[i].astype(float)) ** 2
    )

    psnr_m = cv2.PSNR(
        original,
        mean_images[i]
    )

    ssim_m = ssim(
        original,
        mean_images[i]
    )


    # Median Filter Metrics

    mse_md = np.mean(
        (original.astype(float) -
         median_images[i].astype(float)) ** 2
    )

    psnr_md = cv2.PSNR(
        original,
        median_images[i]
    )

    ssim_md = ssim(
        original,
        median_images[i]
    )


    # Gaussian Filter Metrics

    mse_g = np.mean(
        (original.astype(float) -
         gaussian_images[i].astype(float)) ** 2
    )

    psnr_g = cv2.PSNR(
        original,
        gaussian_images[i]
    )

    ssim_g = ssim(
        original,
        gaussian_images[i]
    )


    mse_mean.append(mse_m)
    mse_median.append(mse_md)
    mse_gaussian.append(mse_g)

    psnr_mean.append(psnr_m)
    psnr_median.append(psnr_md)
    psnr_gaussian.append(psnr_g)

    ssim_mean.append(ssim_m)
    ssim_median.append(ssim_md)
    ssim_gaussian.append(ssim_g)


# ============================================================
# 9. Calculate the Average Values
# ============================================================

average_mse = [
    np.mean(mse_mean),
    np.mean(mse_median),
    np.mean(mse_gaussian)
]

average_psnr = [
    np.mean(psnr_mean),
    np.mean(psnr_median),
    np.mean(psnr_gaussian)
]

average_ssim = [
    np.mean(ssim_mean),
    np.mean(ssim_median),
    np.mean(ssim_gaussian)
]


print("Average Restoration Results")
print("---------------------------")

print("Mean Filter:")
print("MSE   =", round(average_mse[0], 2))
print("PSNR  =", round(average_psnr[0], 2), "dB")
print("SSIM  =", round(average_ssim[0], 4))

print()

print("Median Filter:")
print("MSE   =", round(average_mse[1], 2))
print("PSNR  =", round(average_psnr[1], 2), "dB")
print("SSIM  =", round(average_ssim[1], 4))

print()

print("Gaussian Filter:")
print("MSE   =", round(average_mse[2], 2))
print("PSNR  =", round(average_psnr[2], 2), "dB")
print("SSIM  =", round(average_ssim[2], 4))


# ============================================================
# 10. Restoration Performance Comparison
# ============================================================

filters = [
    "Mean",
    "Median",
    "Gaussian"
]


# MSE Graph

plt.figure(figsize=(7, 5))

plt.bar(filters, average_mse)

plt.title("Average MSE Comparison")

plt.xlabel("Filter")

plt.ylabel("MSE")

plt.show()


# PSNR Graph

plt.figure(figsize=(7, 5))

plt.bar(filters, average_psnr)

plt.title("Average PSNR Comparison")

plt.xlabel("Filter")

plt.ylabel("PSNR (dB)")

plt.show()


# SSIM Graph

plt.figure(figsize=(7, 5))

plt.bar(filters, average_ssim)

plt.title("Average SSIM Comparison")

plt.xlabel("Filter")

plt.ylabel("SSIM")

plt.show()


# ============================================================
# 11. Crack Segmentation using Otsu Thresholding
# ============================================================

otsu_images = []


for median in median_images:

    threshold_value, otsu = cv2.threshold(
        median,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    otsu_images.append(otsu)


plt.figure(figsize=(12, 8))


for i, otsu in enumerate(otsu_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(otsu, cmap="gray")

    plt.title("Otsu Segmentation " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 12. Morphological Operations
# ============================================================

morph_images = []

kernel = np.ones((3, 3), np.uint8)


for otsu in otsu_images:

    opening = cv2.morphologyEx(
        otsu,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    morph_images.append(closing)


plt.figure(figsize=(12, 8))


for i, morph in enumerate(morph_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(morph, cmap="gray")

    plt.title("Morphological Result " + str(i + 1))

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 13. Before & After Comparison
# ============================================================

image_index = 0


plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)

plt.imshow(
    gray_images[image_index],
    cmap="gray"
)

plt.title("Original Image")

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(
    noisy_images[image_index],
    cmap="gray"
)

plt.title("Noisy Image")

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(
    median_images[image_index],
    cmap="gray"
)

plt.title("Median Restored")

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(
    morph_images[image_index],
    cmap="gray"
)

plt.title("Final Segmentation")

plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 14. Final Segmentation of All 7 Images
# ============================================================

plt.figure(figsize=(12, 8))


for i in range(len(morph_images)):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        morph_images[i],
        cmap="gray"
    )

    plt.title(
        "Crack Segmentation " + str(i + 1)
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 15. Segmentation Analysis
# ============================================================

print("Crack Segmentation Analysis")
print("----------------------------")


for i, segmented in enumerate(morph_images):

    crack_pixels = np.sum(
        segmented == 255
    )

    total_pixels = segmented.size

    crack_percentage = (
        crack_pixels / total_pixels
    ) * 100

    print(
        "Image",
        i + 1,
        ": Detected Crack Region =",
        round(crack_percentage, 2),
        "%"
    )


# ============================================================
# 16. Segmentation Comparison
# ============================================================

image_index = 0


plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(
    median_images[image_index],
    cmap="gray"
)

plt.title("Median Restored")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    otsu_images[image_index],
    cmap="gray"
)

plt.title("Otsu Segmentation")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    morph_images[image_index],
    cmap="gray"
)

plt.title("After Morphology")

plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 17. Canny Edge Detection
# ============================================================

canny_images = []


for median in median_images:

    canny = cv2.Canny(
        median,
        50,
        150
    )

    canny_images.append(canny)


plt.figure(figsize=(12, 8))


for i, canny in enumerate(canny_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        canny,
        cmap="gray"
    )

    plt.title(
        "Canny Edge " + str(i + 1)
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 18. Adaptive Thresholding
# ============================================================

adaptive_images = []


for median in median_images:

    adaptive = cv2.adaptiveThreshold(
        median,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    adaptive_images.append(adaptive)


plt.figure(figsize=(12, 8))


for i, adaptive in enumerate(adaptive_images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        adaptive,
        cmap="gray"
    )

    plt.title(
        "Adaptive Threshold " + str(i + 1)
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 19. Segmentation Technique Comparison
# ============================================================

image_index = 0


plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)

plt.imshow(
    median_images[image_index],
    cmap="gray"
)

plt.title("Median Restored")

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(
    otsu_images[image_index],
    cmap="gray"
)

plt.title("Otsu Thresholding")

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(
    canny_images[image_index],
    cmap="gray"
)

plt.title("Canny Edge Detection")

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(
    adaptive_images[image_index],
    cmap="gray"
)

plt.title("Adaptive Thresholding")

plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 20. Restoration Results Table
# ============================================================

print()
print("Restoration Results Table")
print("-------------------------")

print(
    f"{'Filter':<12}"
    f"{'MSE':<12}"
    f"{'PSNR (dB)':<15}"
    f"{'SSIM':<12}"
)

print("-" * 50)


for i in range(3):

    print(
        f"{filters[i]:<12}"
        f"{average_mse[i]:<12.2f}"
        f"{average_psnr[i]:<15.2f}"
        f"{average_ssim[i]:<12.4f}"
    )


# ============================================================
# End of Program
# ============================================================
