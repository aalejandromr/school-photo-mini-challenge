from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np
from io import BytesIO
import asyncio
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor

from app.services.subject_extraction import extract_subject, get_subject_mask
from app.services.shadow_generator import generate_realistic_shadow
from app.utils.image_processing import composite_images, image_to_bytes

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=2)

logger = logging.getLogger(__name__)

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
        logger.info("Received shadow generation request")
        
        # Read image files
        foreground_bytes = await foreground.read()
        background_bytes = await background.read()
        logger.info(f"Read images - foreground: {len(foreground_bytes)} bytes, background: {len(background_bytes)} bytes")
        
        # Validate image formats
        try:
            foreground_img = Image.open(BytesIO(foreground_bytes))
            background_img = Image.open(BytesIO(background_bytes))
            logger.info(f"Image formats validated - foreground: {foreground_img.size}, background: {background_img.size}")
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Extract subject from foreground (run in thread pool to avoid blocking)
        try:
            logger.info("Starting subject extraction...")
            extracted_subject, alpha_mask = await asyncio.get_event_loop().run_in_executor(
                executor, extract_subject, foreground_bytes
            )
            subject_mask = get_subject_mask(extracted_subject)
            logger.info("Subject extraction completed")
        except Exception as e:
            logger.error(f"Failed to extract subject: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to extract subject: {str(e)}")
        
        # Validate that subject was extracted
        if np.sum(subject_mask > 128) == 0:
            logger.warning("No subject detected in foreground image")
            raise HTTPException(status_code=400, detail="No subject detected in foreground image")
        
        # Generate shadow (run in thread pool to avoid blocking)
        try:
            logger.info(f"Generating shadow - angle: {light_angle}°, elevation: {light_elevation}°")
            shadow_mask = await asyncio.get_event_loop().run_in_executor(
                executor,
                generate_realistic_shadow,
                subject_mask,
                light_angle,
                light_elevation
            )
            logger.info("Shadow generation completed")
        except Exception as e:
            logger.error(f"Failed to generate shadow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate shadow: {str(e)}")
        
        # Composite images (run in thread pool to avoid blocking)
        try:
            logger.info("Compositing images...")
            result_image = await asyncio.get_event_loop().run_in_executor(
                executor,
                composite_images,
                background_img,
                shadow_mask,
                extracted_subject
            )
            logger.info("Image compositing completed")
        except Exception as e:
            logger.error(f"Failed to composite images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to composite images: {str(e)}")
        
        # Create debug images
        try:
            logger.info("Creating debug images...")
            
            # 1. composite.png - final output (already have result_image)
            composite_bytes = image_to_bytes(result_image, format="PNG")
            
            # 2. shadow_only.png - shadow on transparent background
            shadow_only = Image.new("RGBA", background_img.size, (0, 0, 0, 0))
            shadow_img = Image.fromarray(shadow_mask.astype(np.uint8)).convert("L")
            shadow_img = shadow_img.resize(background_img.size, Image.Resampling.LANCZOS)
            shadow_alpha = np.array(shadow_img)
            shadow_rgba_array = np.array(shadow_only)
            shadow_rgba_array[:, :, 3] = shadow_alpha
            shadow_only = Image.fromarray(shadow_rgba_array)
            shadow_only_bytes = image_to_bytes(shadow_only, format="PNG")
            
            # 3. mask_debug.png - extracted subject mask visualization
            mask_debug = Image.new("RGBA", extracted_subject.size, (0, 0, 0, 0))
            mask_alpha = subject_mask
            mask_rgba_array = np.array(mask_debug)
            # Make it visible with some color (red tint for visibility)
            mask_rgba_array[:, :, 0] = mask_alpha  # Red channel
            mask_rgba_array[:, :, 3] = mask_alpha  # Alpha channel
            mask_debug = Image.fromarray(mask_rgba_array)
            mask_debug_bytes = image_to_bytes(mask_debug, format="PNG")
            
            logger.info("Debug images created")
        except Exception as e:
            logger.error(f"Failed to create debug images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create debug images: {str(e)}")
        
        # Create ZIP file with all three images
        try:
            logger.info("Creating ZIP file...")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('composite.png', composite_bytes)
                zip_file.writestr('shadow_only.png', shadow_only_bytes)
                zip_file.writestr('mask_debug.png', mask_debug_bytes)
            zip_bytes = zip_buffer.getvalue()
            
            logger.info(f"Returning ZIP file - {len(zip_bytes)} bytes")
            
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=shadow_result.zip"}
            )
        except Exception as e:
            logger.error(f"Failed to create ZIP file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create ZIP file: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
