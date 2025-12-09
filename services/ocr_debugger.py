import logging
import base64
import re
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import pytesseract
from paddleocr import PaddleOCR
import boto3

logger = logging.getLogger(__name__)


class OcrDebugger:
    """Service for debugging OCR on individual slots with intermediate image generation."""
    
    def __init__(self):
        """Initialize OCR engines."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
        logger.info("OcrDebugger initialized")
    
    def debug_single_slot(self, card_index, slot_index, kill_box, screenshot_array, card_boxes):
        """
        Debug OCR for a single slot with full intermediate image visibility.
        
        Args:
            card_index: Card number (1-12)
            slot_index: Slot number (1-4)
            kill_box: Dict with x1, y1, x2, y2 (CARD-RELATIVE padded coordinates from Step 2.5)
            screenshot_array: Numpy array of original full-resolution screenshot (from Step 1)
            card_boxes: Dict mapping card_index to card bounds {x1, y1, x2, y2}
            
        Returns:
            Dict with debug info and intermediate images
        """
        try:
            logger.info(f"[DEBUG] Starting debug for Card {card_index} Slot {slot_index}")
            
            # Use provided screenshot array directly (already full-resolution from Step 1)
            img_height, img_width = screenshot_array.shape[:2]
            
            logger.info(f"[DEBUG] Original image size: {img_width}×{img_height}")
            logger.info(f"[DEBUG] Using padded kill box from step2_5: {kill_box}")
            
            # CRITICAL: Look up card bounding box from Step 1
            card_box = card_boxes.get(card_index)
            if card_box is None:
                error_msg = f"No card bounds found for card_index={card_index}"
                logger.error(f"[DEBUG] {error_msg}")
                raise ValueError(error_msg)
            
            card_x1 = int(card_box["x1"])
            card_y1 = int(card_box["y1"])
            card_x2 = int(card_box["x2"])
            card_y2 = int(card_box["y2"])
            
            logger.info(f"[DEBUG] Card {card_index} bounds: x1={card_x1}, y1={card_y1}, x2={card_x2}, y2={card_y2}")
            
            # Convert CARD-RELATIVE kill_box to SCREENSHOT-ABSOLUTE coordinates
            rel_x1 = int(kill_box["x1"])
            rel_y1 = int(kill_box["y1"])
            rel_x2 = int(kill_box["x2"])
            rel_y2 = int(kill_box["y2"])
            
            abs_x1 = card_x1 + rel_x1
            abs_y1 = card_y1 + rel_y1
            abs_x2 = card_x1 + rel_x2
            abs_y2 = card_y1 + rel_y2
            
            # Clamp absolute coordinates to screenshot bounds
            x1 = max(0, min(img_width, abs_x1))
            y1 = max(0, min(img_height, abs_y1))
            x2 = max(0, min(img_width, abs_x2))
            y2 = max(0, min(img_height, abs_y2))
            
            # Validate crop size
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            # DIAGNOSTIC: Print crop dimensions FIRST
            print(f"[DEBUG] CARD BOUNDS: {card_box}")
            print(f"[DEBUG] RELATIVE KILL BOX: {kill_box}")
            print(f"[DEBUG] ABSOLUTE KILL BOX: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"[DEBUG] CROP DIMENSIONS: height={crop_height}, width={crop_width}")
            print(f"[DEBUG] Expected: height ≈ 25-40px, width ≈ 60-80px")
            
            if crop_width < 20 or crop_height < 20:
                error_msg = f"Kill box too small after conversion: {crop_width}×{crop_height}"
                logger.error(f"[DEBUG] {error_msg}")
                print(f"[DEBUG] ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Crop kill box from screenshot using ABSOLUTE PIXEL COORDINATES
            raw_crop = screenshot_array[y1:y2, x1:x2]
            
            logger.info(f"[DEBUG] Final crop shape: {raw_crop.shape[0]}×{raw_crop.shape[1]}")
            print(f"[DEBUG] ACTUAL CROP SHAPE: {raw_crop.shape}")
            
            # Convert to PIL for base64
            raw_crop_pil = Image.fromarray(raw_crop)
            raw_crop_b64 = self._image_to_base64(raw_crop_pil)
            
            # Upscale crop 3x
            upscaled_crop = cv2.resize(raw_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            upscaled_crop_pil = Image.fromarray(upscaled_crop)
            upscaled_crop_b64 = self._image_to_base64(upscaled_crop_pil)
            
            # Preprocess: Convert to grayscale (raw_crop is BGR from numpy array)
            gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            
            # Denoise
            denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            
            # Convert preprocessed to base64
            preprocessed_pil = Image.fromarray(enhanced)
            preprocessed_b64 = self._image_to_base64(preprocessed_pil)
            
            thresh_pil = Image.fromarray(thresh)
            thresh_b64 = self._image_to_base64(thresh_pil)
            
            denoised_pil = Image.fromarray(denoised)
            denoised_b64 = self._image_to_base64(denoised_pil)
            
            logger.info(f"[DEBUG] Preprocessing complete")
            logger.info(f"[DEBUG] Preprocessing thresholds: CLAHE clipLimit=2.0, threshold=127")
            
            # Tesseract OCR - try on denoised image with digit-specific config
            logger.info(f"[DEBUG] Tesseract input array shape: {raw_crop.shape}")
            # Use denoised image for Tesseract (better for digits)
            # Config: --psm 6 = assume single block of text, -c tessedit_char_whitelist=0123456789
            tesseract_config = '--psm 6 -c tessedit_char_whitelist=0123456789'
            tesseract_raw = pytesseract.image_to_string(denoised_pil, config=tesseract_config)
            logger.info(f"[DEBUG] Tesseract raw output: {repr(tesseract_raw)}")
            print(f"[DEBUG] TESSERACT OUTPUT: {repr(tesseract_raw)}")
            
            # PaddleOCR
            logger.info(f"[DEBUG] PaddleOCR input array shape: {raw_crop.shape}")
            paddle_result = self.paddle_ocr.ocr(raw_crop)
            paddle_raw = str(paddle_result) if paddle_result else ""
            logger.info(f"[DEBUG] PaddleOCR raw output: {repr(paddle_raw[:100])}")  # First 100 chars
            print(f"[DEBUG] PADDLE OUTPUT: {repr(paddle_raw[:150])}")
            
            # Claude verification (disabled for now)
            claude_raw = ""
            logger.info(f"[DEBUG] Claude verification: disabled")
            
            # Parse digit from Tesseract + Paddle only
            parsed_digit = self._extract_digit(tesseract_raw, paddle_raw, claude_raw)
            logger.info(f"[DEBUG] Parsed digit: {parsed_digit}")
            print(f"[DEBUG] PARSED DIGIT: {parsed_digit}")
            
            # Calculate confidence
            confidence = self._calculate_confidence(tesseract_raw, paddle_raw, claude_raw, parsed_digit)
            logger.info(f"[DEBUG] Confidence: {confidence}")
            print(f"[DEBUG] CONFIDENCE: {confidence}")
            
            return {
                "card_index": card_index,
                "slot_index": slot_index,
                "debug": {
                    "raw_crop": raw_crop_b64,
                    "upscaled_crop": upscaled_crop_b64,
                    "preprocessed_crop": preprocessed_b64,
                    "threshold_crop": thresh_b64,
                    "denoised_crop": denoised_b64,
                    "tesseract_raw": tesseract_raw,
                    "paddle_raw": paddle_raw,
                    "claude_raw": claude_raw,
                    "parsed_digit": parsed_digit,
                    "confidence": confidence
                }
            }
        
        except Exception as e:
            logger.exception(f"[DEBUG] Error in debug_single_slot: {str(e)}")
            raise
    
    def _image_to_base64(self, img):
        """Convert PIL Image to base64 string."""
        output = BytesIO()
        img.save(output, format='JPEG', quality=95)
        return base64.b64encode(output.getvalue()).decode('utf-8')
    
    def _call_claude_verify(self, image_pil):
        """Call Claude to verify digit in image."""
        try:
            import base64
            
            # Convert to base64
            output = BytesIO()
            image_pil.save(output, format='JPEG', quality=95)
            image_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
            
            # Call Claude with correct message format
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": base64.b64decode(image_b64)
                                    }
                                }
                            },
                            {
                                "text": "What digit (0-9) do you see in this image? Reply with ONLY the digit or 'NONE' if no digit is visible."
                            }
                        ]
                    }
                ]
            )
            
            return response["output"]["message"]["content"][0]["text"]
        
        except Exception as e:
            logger.error(f"[DEBUG] Claude verification error: {str(e)}")
            return ""
    
    def _extract_digit(self, tesseract_raw, paddle_raw, claude_raw):
        """Extract first digit from OCR outputs."""
        # Try each output in order
        for output in [tesseract_raw, paddle_raw, claude_raw]:
            match = re.search(r'\d', str(output))
            if match:
                digit = int(match.group())
                if 0 <= digit <= 20:
                    return digit
        
        return None
    
    def _calculate_confidence(self, tesseract_raw, paddle_raw, claude_raw, parsed_digit):
        """Calculate confidence based on agreement between engines."""
        if parsed_digit is None:
            return "LOW"
        
        # Count how many engines agree
        agreements = 0
        
        # Tesseract: only count if it actually returned something
        if tesseract_raw and tesseract_raw.strip():
            if re.search(rf'\b{parsed_digit}\b', tesseract_raw):
                agreements += 1
        
        # PaddleOCR: check if digit is in output
        if paddle_raw and re.search(rf'\b{parsed_digit}\b', paddle_raw):
            agreements += 1
        
        # Claude: only if enabled
        if claude_raw and re.search(rf'\b{parsed_digit}\b', claude_raw):
            agreements += 1
        
        # If PaddleOCR detected it, confidence is at least MEDIUM
        if paddle_raw and re.search(rf'\b{parsed_digit}\b', paddle_raw):
            if agreements >= 2:
                return "HIGH"
            else:
                return "MEDIUM"
        
        # Fallback
        if agreements >= 2:
            return "HIGH"
        elif agreements == 1:
            return "MEDIUM"
        else:
            return "LOW"
