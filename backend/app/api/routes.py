from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np
from io import BytesIO

from app.services.subject_extraction import extract_subject, get_subject_mask
from app.services.shadow_generator import generate_realistic_shadow
from app.utils.image_processing import composite_images, image_to_bytes

router = APIRouter()


@router.post("/generate-shadow")
async def generate_shadow(
    foreground: UploadFile = File(..., description="Foreground image with subject"),
    background: UploadFile = File(..., description="Background image"),
    light_angle: float = Form(..., ge=0, le=360, description="Light angle in degrees (0-360)"),
    light_elevation: float = Form(..., ge=0, le=90, description="Light elevation in degrees (0-90)")
):
    """
    Generate realistic shadow composite from foreground and background images.
    
    Args:
        foreground: Image file containing the subject
        background: Background image
        light_angle: Direction of light source (0-360 degrees)
        light_elevation: Elevation of light source (0-90 degrees, 0 = horizontal, 90 = overhead)
        
    Returns:
        Composite image with realistic shadow
    """
    try:
        # Read image files
        foreground_bytes = await foreground.read()
        background_bytes = await background.read()
        
        # Validate image formats
        try:
            foreground_img = Image.open(BytesIO(foreground_bytes))
            background_img = Image.open(BytesIO(background_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Extract subject from foreground
        try:
            extracted_subject, alpha_mask = extract_subject(foreground_bytes)
            subject_mask = get_subject_mask(extracted_subject)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract subject: {str(e)}")
        
        # Validate that subject was extracted
        if np.sum(subject_mask > 128) == 0:
            raise HTTPException(status_code=400, detail="No subject detected in foreground image")
        
        # Generate shadow
        try:
            shadow_mask = generate_realistic_shadow(
                subject_mask=subject_mask,
                light_angle=light_angle,
                light_elevation=light_elevation
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate shadow: {str(e)}")
        
        # Composite images
        try:
            result_image = composite_images(
                background=background_img,
                shadow=shadow_mask,
                foreground=extracted_subject
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to composite images: {str(e)}")
        
        # Convert to bytes and return
        result_bytes = image_to_bytes(result_image, format="PNG")
        
        return Response(content=result_bytes, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
