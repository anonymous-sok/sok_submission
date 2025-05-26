import numpy as np
from io import BytesIO
from PIL import Image as _PILImage
from scipy.ndimage import median_filter as _median_filter
from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
import torch

def _to_uint8(img_arr):
    """
    Take an array in either float [0,1] or uint8 [0,255] and return
    a uint8 copy plus a flag indicating whether we should scale back.
    """
    arr = np.asarray(img_arr)
    if np.issubdtype(arr.dtype, np.floating):
        # assume [0,1]
        arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        was_float = True
    else:
        arr_u8 = arr.astype(np.uint8)
        was_float = False
    return arr_u8, was_float

def _to_original_scale(arr_u8, was_float, orig_dtype):
    """
    Convert back to the original dtype/range.
    """
    if was_float:
        return (arr_u8.astype(np.float32) / 255.).astype(orig_dtype)
    else:
        return arr_u8.astype(orig_dtype)

def _get_image_from_arr(img_arr):
    return _PILImage.fromarray(img_arr, mode='RGB')

def median_filter(img_arr, size=3):
    return _median_filter(img_arr, size=size)

def denoise_tv_bregman(img_arr, weight=30):
    denoised = _denoise_tv_bregman(img_arr, weight=weight) * 255.
    return np.array(denoised, dtype=img_arr.dtype)

def jpeg_compress_pytorch(x, quality=75):
    """
    PyTorch-based JPEG compression using PIL.
    Handles inputs in [0,1] float or [0,255] uint8.
    """
    orig_dtype = x.dtype
    arr_u8, was_float = _to_uint8(x)

    img = _get_image_from_arr(arr_u8)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = _PILImage.open(buffer)
    comp_arr_u8 = np.array(compressed_img)

    return _to_original_scale(comp_arr_u8, was_float, orig_dtype)

def slq(x, qualities=(20, 40, 60, 80), patch_size=8):
    """
    Spatially Localized Quality (SLQ) JPEG compression.
    Handles inputs in [0,1] float or [0,255] uint8.
    """
    orig_dtype = x.dtype
    arr_u8, was_float = _to_uint8(x)
    n, m, _ = arr_u8.shape

    # Divide the image into patches
    patch_n = (n + patch_size - 1) // patch_size
    patch_m = (m + patch_size - 1) // patch_size

    # Random quality for each patch
    random_indices = np.random.randint(0, len(qualities), size=(patch_n, patch_m))
    compressed_image = np.zeros_like(arr_u8)

    for i in range(patch_n):
        for j in range(patch_m):
            sx, sy = i * patch_size, j * patch_size
            ex, ey = min((i + 1) * patch_size, n), min((j + 1) * patch_size, m)
            patch = arr_u8[sx:ex, sy:ey]
            q = qualities[random_indices[i, j]]

            # compress patch
            img = _get_image_from_arr(patch)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            comp_patch = np.array(_PILImage.open(buf))

            compressed_image[sx:ex, sy:ey] = comp_patch

    return _to_original_scale(compressed_image, was_float, orig_dtype)

if __name__ == "__main__":
    # test both float and uint8 inputs
    for dtype in (np.uint8, np.float32):
        rng = np.random.RandomState(0)
        if dtype == np.uint8:
            img = rng.randint(0, 256, (256,256,3), dtype=np.uint8)
        else:
            img = rng.rand(256,256,3).astype(np.float32)
        out = slq(img)
        print(f"Input dtype={dtype.__name__}, output min/max = {out.min():.3f}/{out.max():.3f}")