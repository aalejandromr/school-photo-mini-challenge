from PIL import Image
import numpy as np
import cv2


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
