import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class ImageCropper:
    """Service for cropping images based on bounding boxes."""
    
    @staticmethod
    def crop_image(image_bytes: bytes, x1: int, y1: int, x2: int, y2: int) -> bytes:
        """
        Crop an image based on bounding box coordinates.
        
        Args:
            image_bytes: Raw image bytes
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Cropped image bytes (JPEG)
            
        Raises:
            ValueError: If coordinates are invalid
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError(f"Coordinates out of bounds: ({x1},{y1},{x2},{y2}) for image {width}x{height}")
            
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"Invalid box: x1={x1} >= x2={x2} or y1={y1} >= y2={y2}")
            
            # Crop image
            cropped = img.crop((x1, y1, x2, y2))
            
            # Convert to JPEG bytes
            output = BytesIO()
            cropped.save(output, format='JPEG', quality=95)
            return output.getvalue()
        
        except Exception as e:
            logger.error(f"Image crop error: {str(e)}")
            raise ValueError(f"Failed to crop image: {str(e)}")
