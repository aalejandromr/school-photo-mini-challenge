from PIL import Image
import numpy as np
import cv2
from io import BytesIO


def resize_to_fit(image: Image.Image, target_size: tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image to fit within target size while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        target_size: (width, height) target dimensions
        maintain_aspect: If True, maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if not maintain_aspect:
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    return image


def ensure_same_size(image1: Image.Image, image2: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Ensure two images have the same size by resizing the smaller one.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        
    Returns:
        Tuple of (resized image1, resized image2)
    """
    size1 = image1.size
    size2 = image2.size
    
    if size1 == size2:
        return image1, image2
    
    # Use the larger dimensions
    target_size = (max(size1[0], size2[0]), max(size1[1], size2[1]))
    
    img1_resized = image1.resize(target_size, Image.Resampling.LANCZOS) if size1 != target_size else image1
    img2_resized = image2.resize(target_size, Image.Resampling.LANCZOS) if size2 != target_size else image2
    
    return img1_resized, img2_resized


def composite_images(background: Image.Image, shadow: np.ndarray, foreground: Image.Image) -> Image.Image:
    """
    Composite background, shadow, and foreground images.
    
    Args:
        background: Background PIL Image
        shadow: Shadow as numpy array (grayscale, 0-255)
        foreground: Foreground PIL Image with alpha channel
        
    Returns:
        Composite PIL Image
    """
    # Ensure all images are the same size
    bg, fg = ensure_same_size(background, foreground)
    
    # Convert shadow to PIL Image if needed
    if isinstance(shadow, np.ndarray):
        shadow_img = Image.fromarray(shadow.astype(np.uint8)).convert("L")
        shadow_img = shadow_img.resize(bg.size, Image.Resampling.LANCZOS)
    else:
        shadow_img = shadow
    
    # Convert to RGBA for compositing
    bg_rgba = bg.convert("RGBA")
    fg_rgba = fg.convert("RGBA")
    
    # Create shadow layer (black with alpha from shadow mask)
    shadow_rgba = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    shadow_alpha = np.array(shadow_img)
    shadow_rgba_array = np.array(shadow_rgba)
    shadow_rgba_array[:, :, 3] = shadow_alpha
    shadow_rgba = Image.fromarray(shadow_rgba_array)
    
    # Composite: background + shadow + foreground
    result = Image.alpha_composite(bg_rgba, shadow_rgba)
    result = Image.alpha_composite(result, fg_rgba)
    
    return result.convert("RGB")


def estimate_depth_map(background: Image.Image) -> np.ndarray:
    """
    Estimate depth map from background image using edge detection, 
    gradient analysis, and spatial heuristics.
    
    Args:
        background: Background PIL Image
        
    Returns:
        Depth map as numpy array (same size as background): 
        Higher values = closer objects, lower values = farther objects
        Normalized to 0-255 range
    """
    # Convert PIL Image to numpy array
    bg_array = np.array(background.convert("RGB"))
    h, w = bg_array.shape[:2]
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
    
    # Initialize depth map
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    # 1. Edge-based depth: Vertical edges indicate tall objects (buildings, etc.)
    # Strong vertical edges = likely tall objects = closer to camera (in outdoor scenes)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect vertical edges using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    vertical_edges = cv2.dilate(vertical_edges, vertical_kernel, iterations=2)
    
    # Vertical edges contribute to depth (objects closer)
    depth_map += vertical_edges.astype(float) * 30
    
    # 2. Gradient-based depth: Strong gradients indicate object boundaries
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient
    if np.max(gradient_magnitude) > 0:
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    
    # Strong gradients indicate closer objects
    depth_map += gradient_magnitude.astype(float) * 0.5
    
    # 3. Position-based depth: In outdoor scenes, lower objects typically closer
    # Higher Y coordinate (lower in image) = closer to camera
    y_coords = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
    position_depth = (1.0 - (y_coords.astype(float) / h)) * 100
    depth_map += position_depth
    
    # 4. Size-based depth: Larger blobs/clusters = likely closer
    # Use contours to find objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small noise
            cv2.drawContours(object_mask, [contour], -1, 255, -1)
    
    # Larger objects get more depth (closer)
    object_sizes = cv2.distanceTransform(255 - object_mask, cv2.DIST_L2, 5)
    if np.max(object_sizes) > 0:
        object_sizes = (object_sizes / np.max(object_sizes)) * 50
    depth_map += object_sizes.astype(float)
    
    # 5. Color-based depth: Darker regions often indicate closer objects (shadows/under objects)
    # Bright regions often indicate sky/distance
    bg_hsv = cv2.cvtColor(bg_array, cv2.COLOR_RGB2HSV)
    value = bg_hsv[:, :, 2].astype(float)
    # Invert: darker = closer
    color_depth = (255 - value) * 0.2
    depth_map += color_depth
    
    # Normalize depth map to 0-255 range
    if np.max(depth_map) > 0:
        depth_map = (depth_map / np.max(depth_map)) * 255
    else:
        depth_map = np.zeros((h, w), dtype=np.uint8)
    
    # Smooth the depth map
    depth_map = cv2.GaussianBlur(depth_map.astype(float), (15, 15), 0)
    depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
    
    return depth_map


def detect_ground_regions(background: Image.Image) -> np.ndarray:
    """
    Detect ground vs sky regions in background image using color analysis,
    spatial heuristics, and edge detection.
    
    Args:
        background: Background PIL Image
        
    Returns:
        Binary mask (same size as background): 255 = ground, 0 = sky
    """
    # Convert PIL Image to numpy array
    bg_array = np.array(background.convert("RGB"))
    h, w = bg_array.shape[:2]
    
    # Convert to HSV for better color-based segmentation
    bg_hsv = cv2.cvtColor(bg_array, cv2.COLOR_RGB2HSV)
    hue = bg_hsv[:, :, 0]
    saturation = bg_hsv[:, :, 1]
    value = bg_hsv[:, :, 2]
    
    # Initialize ground mask
    ground_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Color-based detection for sky
    # Sky typically has: blue hue (100-140 in OpenCV HSV, which is 0-180 range, so 50-70)
    # Low saturation (< 50 in 0-255 range, or < 25% in 0-100 range)
    # High brightness/value (> 200 in 0-255 range)
    sky_color_mask = (
        ((hue >= 50) & (hue <= 70)) &  # Blue hue range
        (saturation < 64) &  # Low saturation
        (value > 200)  # High brightness
    )
    
    # Spatial heuristics: weight regions differently
    # Upper third more likely sky, lower third more likely ground
    y_coords = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
    
    # Upper region (top 1/3): more weight for sky
    upper_region = y_coords < (h / 3)
    upper_sky_weight = sky_color_mask.astype(float) * 1.5 * upper_region.astype(float)
    
    # Lower region (bottom 1/3): more weight for ground
    lower_region = y_coords > (2 * h / 3)
    lower_ground_weight = (~sky_color_mask).astype(float) * 1.5 * lower_region.astype(float)
    
    # Middle region: neutral weight
    middle_region = (y_coords >= (h / 3)) & (y_coords <= (2 * h / 3))
    
    # Combine spatial and color information
    # Create probability map: higher values = more likely sky
    sky_probability = np.zeros((h, w), dtype=float)
    sky_probability[upper_region] = upper_sky_weight[upper_region]
    sky_probability[lower_region] = 1.0 - lower_ground_weight[lower_region]
    sky_probability[middle_region] = sky_color_mask[middle_region].astype(float)
    
    # Edge detection to find horizon line
    gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find horizontal edges (potential horizon)
    # Use morphological operations to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    
    # For each column, find the lowest horizontal edge (likely horizon)
    horizon_y_per_column = np.zeros(w, dtype=int)
    for x in range(w):
        edge_pixels = np.where(horizontal_edges[:, x] > 0)[0]
        if len(edge_pixels) > 0:
            # Use the lowest edge point as horizon indicator
            horizon_y_per_column[x] = edge_pixels[-1]
        else:
            # No edge found, use heuristic based on color transition
            horizon_y_per_column[x] = int(2 * h / 3)  # Default to lower third
    
    # Smooth horizon line
    if len(horizon_y_per_column) > 0:
        horizon_y_per_column = cv2.GaussianBlur(
            horizon_y_per_column.reshape(1, -1), (1, 21), 0
        )[0].astype(int)
    
    # Create mask based on horizon: everything below horizon is ground
    for x in range(w):
        horizon_y = min(horizon_y_per_column[x], h - 1)
        ground_mask[horizon_y:, x] = 255
    
    # Refine mask using color probability
    # Areas with high sky probability should be masked out from ground
    # Use a higher threshold to be less aggressive - only mark obvious sky as non-ground
    sky_mask = sky_probability > 0.8
    ground_mask[sky_mask] = 0
    
    # Apply spatial heuristics as fallback
    # Force lower 1/2 to be ground (very confident) - shadows can appear in lower half
    ground_mask[int(h / 2):, :] = 255
    
    # Force upper 1/3 to be sky (very confident)
    ground_mask[:int(h / 3), :] = 0
    
    # Smooth the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply slight blur to smooth transitions
    ground_mask = cv2.GaussianBlur(ground_mask.astype(float), (5, 5), 0).astype(np.uint8)
    ground_mask = (ground_mask > 128).astype(np.uint8) * 255
    
    return ground_mask


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
