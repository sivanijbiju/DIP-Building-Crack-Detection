# Building Crack Detection Using Image Restoration and Segmentation Techniques

## Project Overview

This project focuses on detecting cracks in concrete building surfaces using Digital Image Processing (DIP) techniques.
The project applies image restoration to reduce noise and improve image quality, followed by image segmentation to identify possible crack regions. Different restoration and segmentation techniques are implemented and compared based on their results.

## Objectives

- To identify cracks in concrete building images.
- To convert RGB images into grayscale images for processing.
- To introduce salt-and-pepper noise and study image restoration.
- To apply Mean, Median, and Gaussian filters for noise removal.
- To compare restoration techniques using MSE, PSNR, and SSIM.
- To segment crack regions using Otsu Thresholding.
- To improve segmentation using morphological operations.
- To compare Otsu Thresholding, Canny Edge Detection, and Adaptive Thresholding.
- To analyze the strengths and limitations of the implemented techniques.

## Dataset

The project uses the Concrete Crack Images for Classification dataset from Mendeley Data.

- Dataset: Concrete Crack Images for Classification
- Source: Mendeley Data
- Total Images: 40,000
- Positive Images: 20,000 cracked concrete images
- Negative Images: 20,000 non-cracked concrete images
- Image Size: 227 × 227 pixels
- Image Type: RGB
- Selected Images: 7 positive crack images

For this mini project, seven crack images were selected and renamed as:

crack_01.jpg  
crack_02.jpg  
crack_03.jpg  
crack_04.jpg  
crack_05.jpg  
crack_06.jpg  
crack_07.jpg

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Pillow
- Google Colab

## Methodology

The project follows these main steps:

1. Input crack images
2. Grayscale conversion
3. Salt-and-pepper noise addition
4. Image restoration
5. MSE, PSNR and SSIM evaluation
6. Selection of the best restoration result
7. Otsu Thresholding
8. Morphological opening and closing
9. Crack segmentation
10. Canny Edge Detection
11. Adaptive Thresholding
12. Segmentation technique comparison

## Image Restoration

Three spatial filtering techniques were applied using a 5 × 5 kernel:

### Mean Filter

The Mean Filter is used to smooth the noisy image by averaging the neighboring pixel values.

### Median Filter

The Median Filter is used to remove salt-and-pepper noise while preserving important image structures.

### Gaussian Filter

The Gaussian Filter is used to smooth the image using a Gaussian-weighted neighborhood.

The restoration techniques were evaluated using:

- MSE - Mean Squared Error
- PSNR - Peak Signal-to-Noise Ratio
- SSIM - Structural Similarity Index

### Average Restoration Results

| Filter | MSE | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Mean | 50.13 | 31.13 | 0.7171 |
| Median | 9.97 | 38.16 | 0.9392 |
| Gaussian | 70.71 | 29.64 | 0.6180 |

For the seven selected images with the applied salt-and-pepper noise, the Median Filter produced the best restoration results among the three tested filters.

## Image Segmentation

After image restoration, segmentation techniques were applied to identify possible crack regions.

### Otsu Thresholding

Otsu Thresholding was used to automatically separate foreground and background regions.

### Morphological Processing

The segmented images were improved using:

- Morphological Opening
- Morphological Closing

A 3 × 3 structuring element was used.

### Additional Segmentation Techniques

The project also compares:

- Otsu Thresholding
- Canny Edge Detection
- Adaptive Thresholding

## Evaluation

The restoration stage was quantitatively evaluated using:

- MSE
- PSNR
- SSIM

The segmentation stage was mainly evaluated through visual comparison and analysis of the segmented regions.

The percentage of white pixels in the segmented image was also calculated as an indication of the detected foreground region. This value should not be interpreted as the exact physical percentage of crack area because segmentation may also include shadows, stains, and surface textures.

## Project Structure

DIP-Building-Crack-Detection/

    images/
        crack_01.jpg
        crack_02.jpg
        crack_03.jpg
        crack_04.jpg
        crack_05.jpg
        crack_06.jpg
        crack_07.jpg

    screenshots/
        01_original_crack_images.png
        02_grayscale_images.png
        03_noisy_images.png
        04_mean_filtered_images.png
        05_median_filtered_images.png
        06_gaussian_filtered_images.png
        07_restoration_comparison.png
        08_mse_psnr_ssim_results.png
        09_average_restoration_results.png
        10_otsu_segmentation.png
        11_morphological_results.png
        12_before_after_comparison.png
        13_final_crack_segmentation.png
        14_restoration_performance_graphs_1.png
        15_restoration_performance_graphs_2.png
        16_restoration_performance_graphs_3.png
        17_segmentation_analysis.png
        18_segmentation_comparison.png
        19_canny_edge_detection.png
        20_adaptive_thresholding.png
        21_segmentation_technique_comparison.png

    report/
        DIP_Project_Report.pdf

    source_code.py
    requirements.txt
    README.md


## Results

The `screenshots` folder contains the outputs generated during the project, including:

- Original crack images
- Grayscale images
- Noisy images
- Mean-filtered images
- Median-filtered images
- Gaussian-filtered images
- Restoration comparison
- MSE, PSNR and SSIM results
- Otsu segmentation
- Morphological processing
- Final crack segmentation
- Canny Edge Detection
- Adaptive Thresholding
- Segmentation technique comparison

## Limitations

- Only seven positive crack images were used in this mini project.
- Artificial salt-and-pepper noise was used for restoration evaluation.
- The segmentation process may detect shadows, stains, or surface textures along with cracks.
- Ground-truth crack masks were not available for the selected images, so IoU and Dice scores were not calculated.
- The methods may not perform equally well under different lighting conditions and surface textures.

## Future Scope

Future improvements could include:

- Using a larger and more diverse dataset.
- Using manually annotated ground-truth crack masks.
- Testing additional restoration techniques.
- Applying advanced segmentation methods.
- Optimizing morphological processing parameters.
- Developing a deep-learning-based crack detection system.
- Testing the system on real-time building images.
- Developing a user-friendly application for automatic crack detection.

## Conclusion

The project demonstrates how Digital Image Processing techniques can be used for building crack detection.
Image restoration was first performed to reduce noise, followed by segmentation to identify possible crack regions. Among the tested restoration techniques, the Median Filter produced the best quantitative results for the selected images and applied noise condition. Otsu Thresholding, morphological processing, Canny Edge Detection, and Adaptive Thresholding were then used to study crack segmentation.
The project demonstrates the usefulness of basic image processing techniques for analyzing concrete surface images while also highlighting the limitations of traditional segmentation methods.

## References

1. Mendeley Data. Concrete Crack Images for Classification.
2. OpenCV Documentation.
3. scikit-image Documentation.

