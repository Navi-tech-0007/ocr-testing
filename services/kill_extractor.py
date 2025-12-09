import json
import base64
import logging
import re
from typing import Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
import pytesseract
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class KillExtractionError(Exception):
    """Custom exception for kill extraction errors."""
    pass


class KillExtractor:
    """Service for extracting kill counts from cropped kill number images."""
    
    def __init__(self):
        """Initialize the extractor with OCR engines and Claude client."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        
        # Initialize PaddleOCR for digits
        logger.info("Initializing PaddleOCR...")
        self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
        
        logger.info("KillExtractor initialized")
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
        Extract kill count using Tesseract.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted kill count (0-20) or None
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            
            # Tesseract config for digits only
            config = '--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789'
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
    
    def _verify_with_claude(self, image_base64: str, candidate_kill: int) -> bool:
        """
        Verify kill count with Claude.
        
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
        
        # Convert to base64 for Claude
        image_base64 = self._image_to_base64(image_bytes)
        
        # Try PaddleOCR
        paddle_result = self._extract_paddle_ocr(image_bytes)
        
        # Try Tesseract
        tesseract_result = self._extract_tesseract(image_bytes)
        
        logger.info(f"OCR results - Paddle: {paddle_result}, Tesseract: {tesseract_result}")
        
        # Determine confidence and final result
        kills = None
        confidence = "LOW"
        claude_check = False
        
        # If both agree and are valid
        if paddle_result is not None and tesseract_result is not None:
            if paddle_result == tesseract_result:
                kills = paddle_result
                confidence = "HIGH"
                logger.info(f"Both OCR engines agree: {kills}")
            else:
                # Disagree - use Claude to verify
                logger.info(f"OCR engines disagree, using Claude verification")
                if paddle_result is not None:
                    claude_check = self._verify_with_claude(image_base64, paddle_result)
                    if claude_check:
                        kills = paddle_result
                        confidence = "MEDIUM"
                        logger.info(f"Claude confirmed Paddle result: {kills}")
        
        # If only one OCR succeeded, try Claude
        elif paddle_result is not None:
            claude_check = self._verify_with_claude(image_base64, paddle_result)
            if claude_check:
                kills = paddle_result
                confidence = "MEDIUM"
                logger.info(f"Claude confirmed Paddle result: {kills}")
        
        elif tesseract_result is not None:
            claude_check = self._verify_with_claude(image_base64, tesseract_result)
            if claude_check:
                kills = tesseract_result
                confidence = "MEDIUM"
                logger.info(f"Claude confirmed Tesseract result: {kills}")
        
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
