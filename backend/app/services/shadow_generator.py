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


def warp_shadow_with_depth_mesh(
    shadow_mask: np.ndarray,
    depth_map: np.ndarray,
    light_angle: float,
    light_elevation: float,
    warp_strength: float = 0.3,
    grid_size: int = 20
) -> np.ndarray:
    """
    Warp shadow using mesh-based deformation. Creates a grid of control points
    and deforms each mesh cell based on depth map gradients.
    
    Args:
        shadow_mask: Binary shadow mask
        depth_map: Depth map (0-255, higher = closer objects)
        light_angle: Light angle in degrees (0-360)
        light_elevation: Light elevation in degrees (0-90)
        warp_strength: Strength of warping effect (0.0-1.0)
        grid_size: Size of mesh grid cells (smaller = finer deformation)
        
    Returns:
        Warped shadow mask
    """
    h, w = shadow_mask.shape
    
    # Ensure depth_map matches shadow dimensions
    if depth_map.shape != shadow_mask.shape:
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    angle_rad = angle_to_radians(light_angle)
    depth_normalized = depth_map.astype(float) / 255.0
    
    # Calculate depth gradients (surface normals)
    depth_grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    depth_grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(depth_grad_x**2 + depth_grad_y**2) + 1e-6
    
    # Precompute max_grad_magnitude once (outside loop)
    max_grad_magnitude = np.max(grad_magnitude)
    
    # Normalize gradients to get direction
    grad_x_norm = depth_grad_x / grad_magnitude
    grad_y_norm = depth_grad_y / grad_magnitude
    
    # Calculate perpendicular direction (where shadow flows around edges)
    perp_x = -grad_y_norm
    perp_y = grad_x_norm
    
    # Light direction
    light_dir_x = np.cos(angle_rad)
    light_dir_y = np.sin(angle_rad)
    
    # Create mesh grid of control points
    # Sample at grid_size intervals
    grid_x = np.arange(0, w, grid_size)
    grid_y = np.arange(0, h, grid_size)
    
    # Add edges
    if grid_x[-1] < w - 1:
        grid_x = np.append(grid_x, w - 1)
    if grid_y[-1] < h - 1:
        grid_y = np.append(grid_y, h - 1)
    
    # Create mesh of control points (original positions)
    mesh_x_orig, mesh_y_orig = np.meshgrid(grid_x, grid_y)
    
    # Calculate displacement for each control point based on depth
    mesh_displacement_x = np.zeros_like(mesh_x_orig, dtype=float)
    mesh_displacement_y = np.zeros_like(mesh_y_orig, dtype=float)
    
    max_displacement = warp_strength * 150
    
    for i in range(mesh_x_orig.shape[0]):
        for j in range(mesh_x_orig.shape[1]):
            x = int(mesh_x_orig[i, j])
            y = int(mesh_y_orig[i, j])
            
            if x >= w or y >= h:
                continue
            
            # Get depth and gradient at this control point
            depth_val = depth_normalized[y, x]
            edge_strength = grad_magnitude[y, x] / (max_grad_magnitude + 1e-6)
            
            # Calculate flow direction: blend perpendicular to gradient with light direction
            perp_val_x = perp_x[y, x]
            perp_val_y = perp_y[y, x]
            
            # Blend based on edge strength
            blend = min(edge_strength * 2.0, 1.0)
            flow_x = perp_val_x * blend + light_dir_x * (1 - blend)
            flow_y = perp_val_y * blend + light_dir_y * (1 - blend)
            
            # Calculate displacement magnitude based on depth
            displacement_mag = depth_val * max_displacement * (1 + edge_strength)
            
            mesh_displacement_x[i, j] = flow_x * displacement_mag
            mesh_displacement_y[i, j] = flow_y * displacement_mag
    
    # Create deformed mesh (original + displacement)
    mesh_x_deformed = mesh_x_orig + mesh_displacement_x
    mesh_y_deformed = mesh_y_orig + mesh_displacement_y
    
    # Create full-resolution displacement maps using bilinear interpolation
    # from the mesh control points (using OpenCV resize for interpolation)
    # Resize mesh displacements to full image resolution using bilinear interpolation
    mesh_h, mesh_w = mesh_displacement_x.shape
    
    displacement_x_full = cv2.resize(mesh_displacement_x.astype(np.float32), (w, h), 
                                     interpolation=cv2.INTER_LINEAR)
    displacement_y_full = cv2.resize(mesh_displacement_y.astype(np.float32), (w, h),
                                     interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate meshgrid for remapping
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply displacement using remap
    map_x = (x_coords + displacement_x_full).astype(np.float32)
    map_y = (y_coords + displacement_y_full).astype(np.float32)
    
    # Clamp map values to valid range for cv2.remap
    # OpenCV remap expects values within image bounds or slightly outside
    map_x = np.clip(map_x, -1e6, 1e6).astype(np.float32)
    map_y = np.clip(map_y, -1e6, 1e6).astype(np.float32)
    # Replace NaN/Inf with 0
    map_x = np.nan_to_num(map_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    map_y = np.nan_to_num(map_y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    # Remap shadow using deformed mesh
    warped_shadow = cv2.remap(
        shadow_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped_shadow.astype(np.uint8)


def warp_shadow_with_depth(
    shadow_mask: np.ndarray,
    depth_map: np.ndarray,
    light_angle: float,
    light_elevation: float,
    warp_strength: float = 0.3
) -> np.ndarray:
    """
    Warp shadow based on depth map using mesh-based deformation.
    
    Args:
        shadow_mask: Binary shadow mask
        depth_map: Depth map (0-255, higher = closer objects)
        light_angle: Light angle in degrees (0-360)
        light_elevation: Light elevation in degrees (0-90)
        warp_strength: Strength of warping effect (0.0-1.0)
        
    Returns:
        Warped shadow mask
    """
    # Use mesh-based deformation for better curvature effects
    return warp_shadow_with_depth_mesh(shadow_mask, depth_map, light_angle, light_elevation, warp_strength)


def generate_realistic_shadow(
    subject_mask: np.ndarray,
    light_angle: float,
    light_elevation: float,
    contact_decay: float = 0.1,
    soft_blur_base: int = 3,
    soft_blur_factor: float = 0.5,
    soft_opacity_decay: float = 0.05,
    ground_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None
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
        ground_mask: Optional binary mask of ground regions (255=ground, 0=sky).
                     If provided, shadows will only appear on ground regions.
        depth_map: Optional depth map for shadow warping (0-255, higher = closer objects).
                   If provided, shadow will warp around background objects.
        
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
    
    # Apply depth-based warping if depth map is provided
    if depth_map is not None:
        # Use stronger warp strength for more visible effect - single call only
        combined = warp_shadow_with_depth(combined, depth_map, light_angle, light_elevation, warp_strength=0.8)
        # Re-apply projected mask constraint after warping
        combined = np.minimum(combined, shadow_projected)
    
    # Apply ground mask if provided to prevent shadows in sky regions
    if ground_mask is not None:
        # Ensure ground_mask matches shadow dimensions (resize if needed)
        if ground_mask.shape != combined.shape:
            ground_mask_resized = cv2.resize(ground_mask, (combined.shape[1], combined.shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)
        else:
            ground_mask_resized = ground_mask
        
        # Convert ground_mask to binary (values > 128 = ground)
        ground_binary = (ground_mask_resized > 128).astype(np.uint8) * 255
        
        # Apply mask: zero out shadow pixels that fall on sky regions
        combined = combined.astype(float) * (ground_binary.astype(float) / 255.0)
        combined = combined.astype(np.uint8)
    
    return combined
