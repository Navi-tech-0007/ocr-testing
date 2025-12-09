import json
import base64
import logging
import re
import cv2
from typing import Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
import pytesseract

logger = logging.getLogger(__name__)

class KillExtractionError(Exception):
    """Custom exception for kill extraction errors."""
    pass


class KillExtractor:
    """Service for extracting kill counts from cropped kill number images."""
    
    def __init__(self, use_paddle_debug: bool = False):
        """
        Initialize the extractor with OCR engines and Claude client.
        
        Args:
            use_paddle_debug: If True, load PaddleOCR for debug mode only.
        """
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.use_paddle_debug = use_paddle_debug
        self.paddle_ocr = None
        
        # Load PaddleOCR only if debug mode is enabled
        if self.use_paddle_debug:
            try:
                logger.info("Debug mode: Initializing PaddleOCR...")
                from paddleocr import PaddleOCR
                self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR for debug: {str(e)}")
        
        logger.info(f"KillExtractor initialized (paddle_debug={self.use_paddle_debug})")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load Claude verification prompts."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "step3_system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "step3_user_prompt.txt", "r") as f:
            self.user_prompt_template = f.read().strip()
    
    def _validate_image(self, image_bytes: bytes) -> tuple[int, int]:
        """Validate image and return dimensions."""
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            if width <= 0 or height <= 0:
                raise KillExtractionError("Invalid image dimensions")
            return width, height
        except Exception as e:
            raise KillExtractionError(f"Invalid image: {str(e)}")
    
    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.standard_b64encode(image_bytes).decode("utf-8")
    
    def _preprocess_for_ocr(self, image_bytes: bytes) -> Image.Image:
        """
        Fast preprocessing: convert to grayscale and apply adaptive threshold.
        
        This replaces CLAHE + threshold + denoise with a single fast operation.
        """
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy for OpenCV
        img_np = np.array(img)
        
        # Adaptive threshold (fast, effective for digits)
        # ADAPTIVE_THRESH_MEAN_C: threshold = mean of neighborhood
        # blockSize=11: neighborhood size (must be odd)
        # C=2: constant subtracted from mean
        thresholded = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL
        return Image.fromarray(thresholded)
    
    def _extract_paddle_ocr(self, image_bytes: bytes) -> Optional[int]:
        """
        Extract kill count using PaddleOCR.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted kill count (0-20) or None
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            # Convert PIL Image to numpy array (PaddleOCR requires numpy array)
            img_array = np.array(img)
            result = self.paddle_ocr.ocr(img_array)
            
            if not result or not result[0]:
                logger.warning("PaddleOCR returned empty result")
                return None
            
            # Extract text from all detected regions
            texts = []
            for line in result:
                for word_info in line:
                    text = word_info[1]
                    texts.append(text)
            
            combined_text = "".join(texts).strip()
            logger.debug(f"PaddleOCR raw output: {combined_text}")
            
            # Extract digits only
            digits = re.sub(r'[^0-9]', '', combined_text)
            
            if not digits:
                logger.warning("PaddleOCR: No digits found")
                return None
            
            kill_count = int(digits)
            
            # Validate range
            if 0 <= kill_count <= 20:
                logger.info(f"PaddleOCR extracted: {kill_count}")
                return kill_count
            else:
                logger.warning(f"PaddleOCR value out of range: {kill_count}")
                return None
        
        except Exception as e:
            logger.error(f"PaddleOCR error: {str(e)}")
            return None
    
    def _extract_tesseract(self, image_bytes: bytes) -> Optional[int]:
        """
        Extract kill count using Tesseract on preprocessed image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted kill count (0-20) or None
        """
        try:
            # Preprocess: adaptive threshold (fast, no CLAHE overhead)
            img = self._preprocess_for_ocr(image_bytes)
            
            # Tesseract config optimized for digits
            # --oem 1: Legacy OCR engine (faster)
            # --psm 6: Single uniform block of text (good for digit regions)
            # -c tessedit_char_whitelist: Only recognize digits
            config = '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(img, config=config)
            
            text = text.strip()
            logger.debug(f"Tesseract raw output: {text}")
            
            # Extract digits only
            digits = re.sub(r'[^0-9]', '', text)
            
            if not digits:
                logger.warning("Tesseract: No digits found")
                return None
            
            kill_count = int(digits)
            
            # Validate range
            if 0 <= kill_count <= 20:
                logger.info(f"Tesseract extracted: {kill_count}")
                return kill_count
            else:
                logger.warning(f"Tesseract value out of range: {kill_count}")
                return None
        
        except Exception as e:
            logger.error(f"Tesseract error: {str(e)}")
            return None
    
    def _extract_with_claude(self, image_base64: str) -> Optional[int]:
        """
        Extract kill count directly from image using Claude Vision.
        
        Used as fallback when Tesseract fails.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Extracted kill count (0-20) or None
        """
        try:
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                system=[{"text": self.system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": base64.b64decode(image_base64),
                                    },
                                },
                            },
                            {
                                "text": "Extract the kill count number from this image. Return JSON: {\"kills\": <number>}",
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 100,
                    "temperature": 0,
                    "topP": 1,
                }
            )
            
            # Extract response
            if "output" in response and "message" in response["output"]:
                content_blocks = response["output"]["message"]["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if "text" in block:
                            response_text = block["text"]
                            try:
                                data = json.loads(response_text)
                                kills = data.get("kills")
                                if isinstance(kills, int) and 0 <= kills <= 20:
                                    logger.info(f"Claude extracted: {kills}")
                                    return kills
                            except json.JSONDecodeError:
                                logger.warning(f"Claude returned invalid JSON: {response_text}")
                                return None
            
            logger.warning("No text content in Claude response")
            return None
        
        except Exception as e:
            logger.error(f"Claude extraction error: {str(e)}")
            return None
    
    def _verify_with_claude(self, image_base64: str, candidate_kill: int) -> bool:
        """
        Verify kill count with Claude (kept for backward compatibility).
        
        Args:
            image_base64: Base64-encoded image
            candidate_kill: Candidate kill count to verify
            
        Returns:
            True if Claude confirms match, False otherwise
        """
        try:
            user_prompt = self.user_prompt_template.replace("{{paddle_ocr_result}}", str(candidate_kill))
            
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                system=[{"text": self.system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": base64.b64decode(image_base64),
                                    },
                                },
                            },
                            {
                                "text": user_prompt,
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 200,
                    "temperature": 0,
                    "topP": 1,
                }
            )
            
            # Extract response
            if "output" in response and "message" in response["output"]:
                content_blocks = response["output"]["message"]["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if "text" in block:
                            response_text = block["text"]
                            try:
                                data = json.loads(response_text)
                                match = data.get("match", False)
                                logger.info(f"Claude verification: {match}")
                                return match
                            except json.JSONDecodeError:
                                logger.warning(f"Claude returned invalid JSON: {response_text}")
                                return False
            
            logger.warning("No text content in Claude response")
            return False
        
        except Exception as e:
            logger.error(f"Claude verification error: {str(e)}")
            return False
    
    def extract_kill(self, image_bytes: bytes) -> dict:
        """
        Extract kill count from a cropped kill number image.
        
        PRODUCTION PIPELINE (optimized for speed):
        1. Tesseract on adaptive threshold (primary, ~50-100ms)
        2. Claude Vision verification if needed (fallback, ~500ms)
        
        PaddleOCR is NOT used in production (too slow, ~300-700ms per call).
        It's available only in debug mode.
        
        Args:
            image_bytes: Raw image bytes of kill number crop
            
        Returns:
            Dict with kill count, confidence, and sources
            
        Raises:
            KillExtractionError: If extraction fails
        """
        logger.info(f"Incoming kill extraction request. Image size: {len(image_bytes)} bytes")
        
        # Validate image
        self._validate_image(image_bytes)
        
        # Convert to base64 for Claude (only if needed)
        image_base64 = None
        
        # PRODUCTION: Try Tesseract only (fast, reliable on clean crops)
        tesseract_result = self._extract_tesseract(image_bytes)
        
        logger.info(f"Tesseract result: {tesseract_result}")
        
        # Determine confidence and final result
        kills = None
        confidence = "LOW"
        claude_check = False
        paddle_result = None
        
        # If Tesseract succeeded, trust it (HIGH confidence)
        if tesseract_result is not None:
            kills = tesseract_result
            confidence = "HIGH"
            logger.info(f"Tesseract extracted with HIGH confidence: {kills}")
        else:
            # Tesseract failed, use Claude Vision as fallback
            logger.info("Tesseract failed, using Claude Vision fallback")
            image_base64 = self._image_to_base64(image_bytes)
            
            # Try to extract with Claude
            claude_result = self._extract_with_claude(image_base64)
            if claude_result is not None:
                kills = claude_result
                confidence = "MEDIUM"
                claude_check = True
                logger.info(f"Claude extracted with MEDIUM confidence: {kills}")
        
        logger.info(f"Final result - Kills: {kills}, Confidence: {confidence}")
        
        return {
            "kills": kills,
            "confidence": confidence,
            "sources": {
                "paddle": paddle_result,
                "tesseract": tesseract_result,
                "claude_check": claude_check
            }
        }
    
    def extract_kills_batch(self, kill_crops: list[dict]) -> list[dict]:
        """
        Extract kill counts from multiple cropped images.
        
        Args:
            kill_crops: List of dicts with card_index, slot_index, image (base64)
            
        Returns:
            List of extraction results
        """
        results = []
        
        for crop in kill_crops:
            try:
                card_index = crop.get("card_index")
                slot_index = crop.get("slot_index")
                image_base64 = crop.get("image")
                
                if not image_base64:
                    logger.error(f"Card {card_index} Slot {slot_index}: No image provided")
                    results.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "kills": None,
                        "confidence": "LOW",
                        "error": "No image provided",
                        "sources": {}
                    })
                    continue
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(image_base64)
                
                # Extract kill
                extraction = self.extract_kill(image_bytes)
                
                results.append({
                    "card_index": card_index,
                    "slot_index": slot_index,
                    "kills": extraction["kills"],
                    "confidence": extraction["confidence"],
                    "sources": extraction["sources"]
                })
            
            except Exception as e:
                logger.error(f"Error extracting kills for card {crop.get('card_index')} slot {crop.get('slot_index')}: {str(e)}")
                results.append({
                    "card_index": crop.get("card_index"),
                    "slot_index": crop.get("slot_index"),
                    "kills": None,
                    "confidence": "LOW",
                    "error": str(e),
                    "sources": {}
                })
        
        return results
