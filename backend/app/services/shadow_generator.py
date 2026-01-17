import numpy as np
import cv2
from PIL import Image
from typing import Optional


def angle_to_radians(angle: float) -> float:
    """Convert angle in degrees to radians."""
    return np.deg2rad(angle)


def project_shadow_silhouette(
    mask: np.ndarray,
    light_angle: float,
    light_elevation: float,
    shadow_scale: float = 1.0
) -> np.ndarray:
    """
    Project subject silhouette to create shadow shape based on light direction.
    
    Args:
        mask: Binary mask of subject (0 = background, 255 = subject)
        light_angle: Light angle in degrees (0-360)
        light_elevation: Light elevation in degrees (0-90)
        shadow_scale: Scale factor for shadow size
        
    Returns:
        Projected shadow mask
    """
    # Convert to radians
    angle_rad = angle_to_radians(light_angle)
    elevation_rad = angle_to_radians(light_elevation)
    
    # Calculate shadow offset based on elevation
    # Lower elevation = longer shadow
    shadow_length = np.tan(elevation_rad) * mask.shape[0] * 0.5
    
    # Calculate offset in x and y based on angle
    offset_x = shadow_length * np.cos(angle_rad) * shadow_scale
    offset_y = shadow_length * np.sin(angle_rad) * shadow_scale
    
    # Create transformation matrix for perspective/affine transform
    # Scale shadow based on elevation (lower elevation = more stretched)
    scale_factor = 1.0 / np.cos(elevation_rad) if elevation_rad > 0 else 1.0
    
    # Create larger canvas to accommodate shadow
    h, w = mask.shape
    canvas_size = (int(w * 2), int(h * 2))
    shadow_canvas = np.zeros(canvas_size, dtype=np.uint8)
    
    # Calculate center positions
    center_x, center_y = w // 2, h // 2
    canvas_center_x, canvas_center_y = canvas_size[0] // 2, canvas_size[1] // 2
    
    # Place original mask on canvas
    start_x = canvas_center_x - center_x
    start_y = canvas_center_y - center_y
    shadow_canvas[start_y:start_y+h, start_x:start_x+w] = mask
    
    # Create affine transformation matrix
    # Scale in the direction of light
    M = cv2.getRotationMatrix2D((canvas_center_x, canvas_center_y), 0, scale_factor)
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    
    # Apply transformation
    shadow_projected = cv2.warpAffine(
        shadow_canvas,
        M,
        canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Crop back to original size (centered)
    crop_x = (canvas_size[0] - w) // 2
    crop_y = (canvas_size[1] - h) // 2
    shadow_projected = shadow_projected[crop_y:crop_y+h, crop_x:crop_x+w]
    
    return shadow_projected


def find_contact_points(mask: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Find contact points at the bottom of the subject.
    
    Args:
        mask: Binary mask of subject
        threshold: Minimum distance from bottom to consider as contact
        
    Returns:
        Binary mask of contact points
    """
    h, w = mask.shape
    contact_mask = np.zeros_like(mask)
    
    # Find bottom-most points of the subject
    for x in range(w):
        # Find the lowest point with subject in this column
        for y in range(h - 1, -1, -1):
            if mask[y, x] > 128:  # Subject pixel
                # Mark contact area (bottom few pixels)
                contact_y_start = max(0, y - threshold)
                contact_mask[contact_y_start:y+1, x] = 255
                break
    
    return contact_mask


def generate_contact_shadow(
    shadow_mask: np.ndarray,
    contact_points: np.ndarray,
    decay_factor: float = 0.1
) -> np.ndarray:
    """
    Generate contact shadow with exponential falloff from contact points.
    
    Args:
        shadow_mask: Projected shadow mask
        contact_points: Binary mask of contact points
        decay_factor: Decay factor for opacity falloff
        
    Returns:
        Contact shadow with opacity falloff
    """
    # Calculate distance from contact points
    # Invert contact points for distance transform (distance from non-contact)
    contact_inv = 255 - contact_points
    dist_transform = cv2.distanceTransform(contact_inv, cv2.DIST_L2, 5)
    
    # Normalize distance
    max_dist = np.max(dist_transform)
    if max_dist > 0:
        dist_normalized = dist_transform / max_dist
    else:
        dist_normalized = dist_transform
    
    # Apply exponential decay: opacity = exp(-distance * decay_factor)
    opacity = np.exp(-dist_normalized * decay_factor * 10)
    
    # Apply opacity to shadow mask
    contact_shadow = (shadow_mask.astype(float) * opacity).astype(np.uint8)
    
    # Ensure contact area is dark
    contact_shadow[contact_points > 128] = 255
    
    return contact_shadow


def generate_soft_shadow(
    shadow_mask: np.ndarray,
    contact_points: np.ndarray,
    base_blur: int = 3,
    blur_factor: float = 0.5,
    opacity_decay: float = 0.05
) -> np.ndarray:
    """
    Generate soft shadow with distance-based blur and opacity.
    
    Args:
        shadow_mask: Projected shadow mask
        contact_points: Binary mask of contact points
        base_blur: Base blur radius
        blur_factor: Blur increase per distance unit
        opacity_decay: Opacity decay factor
        
    Returns:
        Soft shadow with variable blur and opacity
    """
    # Calculate distance from contact points
    contact_inv = 255 - contact_points
    dist_transform = cv2.distanceTransform(contact_inv, cv2.DIST_L2, 5)
    
    # Normalize distance
    max_dist = np.max(dist_transform)
    if max_dist == 0:
        return shadow_mask
    
    dist_normalized = dist_transform / max_dist
    
    # Calculate maximum blur radius - cap it to prevent excessive computation
    # For large images, limit blur to reasonable values
    calculated_max_blur = int(base_blur + max_dist * blur_factor)
    max_blur = min(calculated_max_blur, 101)  # Cap at 101 (reasonable max for large images)
    
    # Apply blur in discrete steps for efficiency (instead of per-pixel)
    # Use fewer blur levels to reduce computation
    num_blur_levels = min(10, max_blur // 2 + 1)  # At most 10 blur levels
    blur_levels = np.linspace(1, max_blur, num_blur_levels, dtype=int)
    # Ensure odd numbers only for GaussianBlur
    blur_levels = [(b + 1) if b % 2 == 0 else b for b in blur_levels]
    blur_levels = sorted(list(set(blur_levels)))  # Remove duplicates and sort
    
    h, w = shadow_mask.shape
    soft_shadow = np.zeros_like(shadow_mask, dtype=np.float32)
    
    # Apply blur at discrete levels
    prev_threshold = 0.0
    for i, blur_radius in enumerate(blur_levels):
        if blur_radius < 1:
            continue
            
        # Determine distance threshold for this blur level
        if i == len(blur_levels) - 1:
            # Last level: everything beyond previous threshold
            threshold = 1.0
        else:
            threshold = (i + 1) / len(blur_levels)
        
        # Create mask for pixels in this distance range
        region_mask = (dist_normalized >= prev_threshold) & (dist_normalized < threshold)
        
        if np.any(region_mask):
            # Apply blur once for this level
            kernel_size = blur_radius * 2 + 1
            if kernel_size > 0:
                blurred = cv2.GaussianBlur(shadow_mask, (kernel_size, kernel_size), 0)
                soft_shadow[region_mask] = blurred[region_mask].astype(float)
        
        prev_threshold = threshold
    
    # Apply opacity decay with distance
    opacity = np.exp(-dist_normalized * opacity_decay * 20)
    soft_shadow = (soft_shadow * opacity).astype(np.uint8)
    
    # Ensure contact area remains sharp
    soft_shadow[contact_points > 128] = shadow_mask[contact_points > 128]
    
    return soft_shadow


def generate_realistic_shadow(
    subject_mask: np.ndarray,
    light_angle: float,
    light_elevation: float,
    contact_decay: float = 0.1,
    soft_blur_base: int = 3,
    soft_blur_factor: float = 0.5,
    soft_opacity_decay: float = 0.05
) -> np.ndarray:
    """
    Generate complete realistic shadow with contact and soft components.
    
    Args:
        subject_mask: Binary mask of subject
        light_angle: Light angle in degrees (0-360)
        light_elevation: Light elevation in degrees (0-90)
        contact_decay: Decay factor for contact shadow
        soft_blur_base: Base blur radius for soft shadow
        soft_blur_factor: Blur increase factor
        soft_opacity_decay: Opacity decay factor for soft shadow
        
    Returns:
        Complete shadow mask with realistic falloff
    """
    # Project shadow silhouette
    shadow_projected = project_shadow_silhouette(subject_mask, light_angle, light_elevation)
    
    # Find contact points
    contact_points = find_contact_points(subject_mask)
    
    # Project contact points to shadow position
    angle_rad = angle_to_radians(light_angle)
    elevation_rad = angle_to_radians(light_elevation)
    shadow_length = np.tan(elevation_rad) * subject_mask.shape[0] * 0.5 if elevation_rad > 0 else 0
    offset_x = int(shadow_length * np.cos(angle_rad))
    offset_y = int(shadow_length * np.sin(angle_rad))
    
    # Translate contact points
    h, w = contact_points.shape
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    contact_projected = cv2.warpAffine(contact_points, M, (w, h), borderValue=0)
    
    # Generate contact shadow
    contact_shadow = generate_contact_shadow(shadow_projected, contact_projected, contact_decay)
    
    # Generate soft shadow
    soft_shadow = generate_soft_shadow(
        shadow_projected,
        contact_projected,
        soft_blur_base,
        soft_blur_factor,
        soft_opacity_decay
    )
    
    # Combine contact and soft shadows
    # Contact shadow provides sharpness near contact, soft shadow provides smooth falloff
    combined = np.maximum(contact_shadow, soft_shadow)
    
    # Ensure shadow only exists where projected mask exists
    combined = np.minimum(combined, shadow_projected)
    
    return combined
