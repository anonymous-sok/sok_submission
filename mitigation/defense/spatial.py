"""
A spatial smoothing defense that implements both local smoothing and non-local smoothing

we follow the design of this paper:

https://arxiv.org/pdf/1704.01155


"""

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def local_spatial_smoothing(img, kernel_size=3):    
    """
    perform median filtering on the input image
    Args:
        img: np.ndarray, input image (can be grayscale or color, value range 0-1)
        kernel_size: int, size of the filtering window
    Returns:
        np.ndarray: image after median filtering
    """
    if img.ndim == 2:  # single channel
        return ndimage.median_filter(img, size=kernel_size, mode='reflect')
    elif img.ndim == 3:  # multi-channel
        return np.stack([ndimage.median_filter(img[..., c], size=kernel_size, mode='reflect')
                         for c in range(img.shape[-1])], axis=-1)
    else:
        raise ValueError('Unsupported image dimension!')
    
def to_uint8_safe(img):

    if img.dtype == np.uint8:
        return img         
    elif np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:          
            img = img * 255.0
        
        return np.clip(img, 0, 255).astype(np.uint8)
    elif np.issubdtype(img.dtype, np.integer):
        if img.min() < 0 or img.max() > 255:
            raise ValueError("range exceed 0-255")
        return img.astype(np.uint8)
    else:
        raise TypeError("unsupported dtype")
    
def non_local_spatial_smoothing(img, h=10, templateWindowSize=7, searchWindowSize=21):

    if img.max() <= 1.0:
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        
    if img_uint8.ndim == 2:
        result = cv2.fastNlMeansDenoising(img_uint8, None, h, templateWindowSize, searchWindowSize)
    elif img_uint8.ndim == 3:
        result = cv2.fastNlMeansDenoisingColored(img_uint8, None, h, h, templateWindowSize, searchWindowSize)
    else:
        raise ValueError('Unsupported image dimension!')
    
    if img.max() <= 1.0:
        return result.astype(np.float32) / 255.0
    else:
        return result.astype(np.uint8)


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    output_dir = './spatial_out'
    os.makedirs(output_dir, exist_ok=True)

    gray = np.random.rand(128, 128)             # 1 channel, 0~1
    color = np.random.rand(128, 128, 3)         # 3 channels, 0~1

    # Local Smoothing: median filter
    gray_local = local_spatial_smoothing(gray, kernel_size=3)
    color_local = local_spatial_smoothing(color, kernel_size=5)

    # Non-local Smoothing: non-local means
    gray_non_local = non_local_spatial_smoothing(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    color_non_local = non_local_spatial_smoothing(color, h=15, templateWindowSize=7, searchWindowSize=21)

    # save images
    def save_image(img, path, cmap=None):
        plt.imsave(path, img, cmap=cmap)

    # save grayscale images
    save_image(gray,        os.path.join(output_dir, 'gray_noisy.png'),        cmap='gray')
    save_image(gray_local,        os.path.join(output_dir, 'gray_local.png'),        cmap='gray')
    save_image(gray_non_local,    os.path.join(output_dir, 'gray_non_local.png'),    cmap='gray')

    # save color images
    save_image(color,             os.path.join(output_dir, 'color.png'))
    save_image(color_local,       os.path.join(output_dir, 'color_local.png'))
    save_image(color_non_local,   os.path.join(output_dir, 'color_non_local.png'))

    print(f"Saved all images to {output_dir}/")