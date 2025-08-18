import numpy as np
import torch
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' format if needed."""
    if input_order == 'CHW':
        return img.transpose(1, 2, 0)
    else:
        return img


@METRIC_REGISTRY.register()
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


@METRIC_REGISTRY.register()
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


@METRIC_REGISTRY.register()
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


@METRIC_REGISTRY.register()
def calculate_sam_pt(img, img2, crop_border, **kwargs):
    """Calculate SAM (Spectral Angle Mapper) for hyperspectral images (PyTorch version).
    
    Args:
        img (Tensor): Reference images with range [0, 1], shape (n, c, h, w).
        img2 (Tensor): Test images with range [0, 1], shape (n, c, h, w).
        crop_border (int): Cropped pixels in each edge of an image.
        
    Returns:
        Tensor: SAM result in radians.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)
    
    # Reshape to (batch_size, num_bands, num_pixels)
    batch_size, num_bands, height, width = img.shape
    img_flat = img.view(batch_size, num_bands, -1)
    img2_flat = img2.view(batch_size, num_bands, -1)
    
    # Calculate spectral angle for each pixel
    dot_product = torch.sum(img_flat * img2_flat, dim=1)  # (batch_size, num_pixels)
    norm_img = torch.norm(img_flat, dim=1)
    norm_img2 = torch.norm(img2_flat, dim=1)
    
    # Avoid division by zero
    valid_pixels = (norm_img > 0) & (norm_img2 > 0)
    
    cos_angle = torch.where(
        valid_pixels,
        dot_product / (norm_img * norm_img2 + 1e-8),
        torch.zeros_like(dot_product)
    )
    
    # Clamp to avoid numerical errors
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    angles = torch.acos(cos_angle)
    
    # Calculate mean only for valid pixels
    sam_values = []
    for b in range(batch_size):
        valid_mask = valid_pixels[b]
        if torch.sum(valid_mask) > 0:
            sam_values.append(torch.mean(angles[b, valid_mask]))
        else:
            sam_values.append(torch.tensor(0.0, device=img.device))
    
    return torch.stack(sam_values)


@METRIC_REGISTRY.register()
def calculate_ergas_pt(img, img2, crop_border, scale=1, **kwargs):
    """Calculate ERGAS (PyTorch version).
    
    Args:
        img (Tensor): Reference images with range [0, 1], shape (n, c, h, w).
        img2 (Tensor): Test images with range [0, 1], shape (n, c, h, w).
        crop_border (int): Cropped pixels in each edge of an image.
        scale (int): Scale factor for super-resolution. Default: 1.
        
    Returns:
        Tensor: ERGAS result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    img = img.to(torch.float64) * 255.0  # Convert to [0, 255] range
    img2 = img2.to(torch.float64) * 255.0
    
    batch_size, num_bands = img.shape[:2]
    
    sum_squared_relative_error = torch.zeros(batch_size, device=img.device, dtype=torch.float64)
    
    for i in range(num_bands):
        band_ref = img[:, i, :, :]  # (batch_size, h, w)
        band_test = img2[:, i, :, :]
        
        mean_ref = torch.mean(band_ref, dim=[1, 2])  # (batch_size,)
        mse_band = torch.mean((band_ref - band_test) ** 2, dim=[1, 2])
        
        # Avoid division by zero
        valid_bands = mean_ref != 0
        relative_error = torch.where(
            valid_bands,
            mse_band / (mean_ref ** 2 + 1e-8),
            torch.zeros_like(mse_band)
        )
        sum_squared_relative_error += relative_error
    
    ergas = 100 * scale * torch.sqrt(sum_squared_relative_error / num_bands)
    return ergas


@METRIC_REGISTRY.register()
def calculate_rmse_pt(img, img2, crop_border, **kwargs):
    """Calculate RMSE (PyTorch version).
    
    Args:
        img (Tensor): Reference images with range [0, 1], shape (n, c, h, w).
        img2 (Tensor): Test images with range [0, 1], shape (n, c, h, w).
        crop_border (int): Cropped pixels in each edge of an image.
        
    Returns:
        Tensor: RMSE result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    img = img.to(torch.float64) * 255.0  # Convert to [0, 255] range
    img2 = img2.to(torch.float64) * 255.0
    
    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    rmse = torch.sqrt(mse)
    return rmse