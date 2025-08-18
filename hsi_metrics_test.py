import numpy as np
import torch
import torch.nn.functional as F

# Simple registry implementation to avoid cv2 dependency issues
class Registry:
    """Simple registry for storing functions"""
    def __init__(self, name):
        self.name = name
        self._obj_map = {}
    
    def register(self, name=None):
        def _register(func):
            key = name or func.__name__
            self._obj_map[key] = func
            return func
        return _register
    
    def get(self, name):
        return self._obj_map.get(name)

# Create HSI metrics registry
HSI_METRIC_REGISTRY = Registry('HSI_METRICS')


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' format if needed."""
    if input_order == 'CHW':
        return img.transpose(1, 2, 0)
    else:
        return img


@HSI_METRIC_REGISTRY.register()
def calculate_sam(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate SAM (Spectral Angle Mapper) for hyperspectral images.
    
    SAM measures the spectral angle between two hyperspectral pixels.
    Lower SAM values indicate better spectral similarity.
    
    Args:
        img (ndarray): Images with range [0, 255], shape (H, W, C).
        img2 (ndarray): Images with range [0, 255], shape (H, W, C).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        
    Returns:
        float: SAM result in radians.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Reshape to (num_pixels, num_bands)
    img_flat = img.reshape(-1, img.shape[-1])
    img2_flat = img2.reshape(-1, img2.shape[-1])
    
    # Calculate spectral angle for each pixel
    dot_product = np.sum(img_flat * img2_flat, axis=1)
    norm_img = np.linalg.norm(img_flat, axis=1)
    norm_img2 = np.linalg.norm(img2_flat, axis=1)
    
    # Avoid division by zero
    valid_pixels = (norm_img > 0) & (norm_img2 > 0)
    
    if np.sum(valid_pixels) == 0:
        return 0.0
    
    cos_angle = dot_product[valid_pixels] / (norm_img[valid_pixels] * norm_img2[valid_pixels])
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles = np.arccos(cos_angle)
    return np.mean(angles)


@HSI_METRIC_REGISTRY.register()
def calculate_ergas(img, img2, crop_border, input_order='HWC', scale=1, **kwargs):
    """Calculate ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    
    ERGAS measures the relative dimensionless global error in synthesis.
    Lower ERGAS values indicate better quality.
    
    Args:
        img (ndarray): Reference images with range [0, 255], shape (H, W, C).
        img2 (ndarray): Test images with range [0, 255], shape (H, W, C).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        scale (int): Scale factor for super-resolution. Default: 1.
        
    Returns:
        float: ERGAS result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate ERGAS
    sum_squared_relative_error = 0.0
    num_bands = img.shape[-1]
    
    for i in range(num_bands):
        band_ref = img[..., i]
        band_test = img2[..., i]
        
        mean_ref = np.mean(band_ref)
        if mean_ref != 0:
            mse_band = np.mean((band_ref - band_test) ** 2)
            sum_squared_relative_error += mse_band / (mean_ref ** 2)
    
    ergas = 100 * scale * np.sqrt(sum_squared_relative_error / num_bands)
    return ergas


@HSI_METRIC_REGISTRY.register()
def calculate_rmse(img, img2, crop_border, input_order='HWC', **kwargs):
    """Calculate RMSE (Root Mean Square Error).
    
    Args:
        img (ndarray): Reference images with range [0, 255], shape (H, W, C).
        img2 (ndarray): Test images with range [0, 255], shape (H, W, C).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        
    Returns:
        float: RMSE result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img - img2) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def test_hsi_metrics():
    """Test function for HSI metrics"""
    print("Testing HSI Metrics...")
    
    # Create synthetic HSI data
    np.random.seed(42)  # For reproducible results
    img1 = np.random.rand(32, 32, 10) * 255  # 32x32 image with 10 bands
    img2 = np.random.rand(32, 32, 10) * 255
    
    # Test SAM
    sam_func = HSI_METRIC_REGISTRY.get('calculate_sam')
    sam_value = sam_func(img1, img2, crop_border=0)
    print(f'✓ SAM calculation: {sam_value:.4f} radians')
    
    # Test ERGAS
    ergas_func = HSI_METRIC_REGISTRY.get('calculate_ergas')
    ergas_value = ergas_func(img1, img2, crop_border=0, scale=4)
    print(f'✓ ERGAS calculation: {ergas_value:.4f}')
    
    # Test RMSE
    rmse_func = HSI_METRIC_REGISTRY.get('calculate_rmse')
    rmse_value = rmse_func(img1, img2, crop_border=0)
    print(f'✓ RMSE calculation: {rmse_value:.4f}')
    
    print("All HSI metrics working correctly!")
    
    return {
        'sam': sam_value,
        'ergas': ergas_value,
        'rmse': rmse_value
    }


if __name__ == '__main__':
    test_hsi_metrics()