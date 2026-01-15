from PIL import Image
import numpy as np
from rembg import remove
from io import BytesIO


def extract_subject(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """
    Extract subject from foreground image using rembg.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Tuple of (RGBA image with transparent background, alpha mask as numpy array)
    """
    # Remove background using rembg
    output_bytes = remove(image_bytes)
    
    # Convert to PIL Image
    extracted_image = Image.open(BytesIO(output_bytes)).convert("RGBA")
    
    # Extract alpha channel as mask
    alpha_channel = np.array(extracted_image.split()[3])
    
    return extracted_image, alpha_channel


def get_subject_mask(image: Image.Image) -> np.ndarray:
    """
    Get binary mask from RGBA image.
    
    Args:
        image: PIL Image with alpha channel
        
    Returns:
        Binary mask (0 = background, 255 = subject)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    alpha = np.array(image.split()[3])
    return alpha
