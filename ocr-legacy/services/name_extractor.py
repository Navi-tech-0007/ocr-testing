import json
import base64
import logging
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import boto3

logger = logging.getLogger(__name__)

class NameExtractionError(Exception):
    """Custom exception for name extraction errors."""
    pass


class NameExtractor:
    """Service for extracting player names from cropped name images."""
    
    def __init__(self):
        """Initialize the extractor with Claude client."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        logger.info("NameExtractor initialized")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load Claude prompts from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "step4_system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "step4_user_prompt.txt", "r") as f:
            self.user_prompt = f.read().strip()
        
        with open(prompts_dir / "step4_retry_prompt.txt", "r") as f:
            self.retry_prompt = f.read().strip()
    
    def _validate_image(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Validate image and return dimensions.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            NameExtractionError: If image is invalid
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            if width <= 0 or height <= 0:
                raise NameExtractionError("Invalid image dimensions")
            return width, height
        except Exception as e:
            raise NameExtractionError(f"Invalid image: {str(e)}")
    
    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.standard_b64encode(image_bytes).decode("utf-8")
    
    def _call_claude_vision(self, image_base64: str, prompt_text: str, is_retry: bool = False) -> str:
        """
        Call Claude 3.7 Sonnet Vision via AWS Bedrock with image and prompt.
        
        Args:
            image_base64: Base64-encoded image
            prompt_text: The user prompt to send
            is_retry: Whether this is a retry attempt
            
        Returns:
            Raw response text from Claude
        """
        response = self.bedrock_client.converse(
            modelId=self.model_id,
            system=[
                {
                    "text": self.system_prompt
                }
            ],
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
                            "text": prompt_text,
                        }
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": 1500,
                "temperature": 0,
                "topP": 1,
            }
        )
        
        if "output" in response and "message" in response["output"]:
            content_blocks = response["output"]["message"]["content"]
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if "text" in block:
                        return block["text"]
        
        raise NameExtractionError("No text content in Bedrock response")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from Claude.
        
        Args:
            response_text: Raw response text from Claude
            
        Returns:
            Parsed JSON dict
            
        Raises:
            NameExtractionError: If JSON is invalid
        """
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(response_text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        
        raise NameExtractionError("Invalid JSON response from Claude")
    
    def _validate_name_schema(self, data: dict) -> str:
        """
        Validate name schema and extract name string.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Extracted name string (may be empty)
            
        Raises:
            NameExtractionError: If schema is invalid
        """
        if not isinstance(data, dict):
            raise NameExtractionError("Response must be a JSON object")
        
        if "name" not in data:
            raise NameExtractionError("Response must contain 'name' key")
        
        name = data.get("name")
        
        if not isinstance(name, str):
            raise NameExtractionError(f"'name' must be a string, got {type(name).__name__}")
        
        # Return the exact string as-is (no modifications)
        return name
    
    def extract_name(self, image_bytes: bytes) -> dict:
        """
        Extract player name from a cropped name image.
        
        Args:
            image_bytes: Raw image bytes of name crop
            
        Returns:
            Dict with name and confidence
            
        Raises:
            NameExtractionError: If extraction fails
        """
        logger.info(f"Incoming name extraction request. Image size: {len(image_bytes)} bytes")
        
        # Validate image
        self._validate_image(image_bytes)
        
        # Convert to base64
        image_base64 = self._image_to_base64(image_bytes)
        
        # Call Claude Vision
        start_time = time.time()
        response_text = self._call_claude_vision(image_base64, self.user_prompt, is_retry=False)
        latency = time.time() - start_time
        logger.info(f"Bedrock Claude call completed. Latency: {latency:.2f}s")
        
        try:
            data = self._parse_json_response(response_text)
            logger.debug("JSON parsed successfully on first attempt")
        except NameExtractionError as e:
            logger.warning(f"JSON parse failed on first attempt: {str(e)}. Retrying...")
            start_time = time.time()
            response_text = self._call_claude_vision(image_base64, self.retry_prompt, is_retry=True)
            latency = time.time() - start_time
            logger.info(f"Bedrock Claude retry call completed. Latency: {latency:.2f}s")
            try:
                data = self._parse_json_response(response_text)
                logger.info("JSON parsed successfully on retry")
            except NameExtractionError as e:
                logger.error(f"JSON parse failed after retry: {str(e)}")
                raise NameExtractionError(f"Failed to parse JSON after retry: {str(e)}")
        
        # Validate schema and extract name
        name = self._validate_name_schema(data)
        
        # Determine confidence based on whether name is empty
        confidence = "LOW" if name == "" else "HIGH"
        
        logger.info(f"Name extraction successful. Name: '{name}', Confidence: {confidence}")
        
        return {
            "name": name,
            "confidence": confidence
        }
    
    def extract_names_batch(self, name_crops: list[dict]) -> list[dict]:
        """
        Extract player names from multiple cropped images.
        
        Args:
            name_crops: List of dicts with card_index, slot_index, image (base64)
            
        Returns:
            List of extraction results
        """
        results = []
        
        for crop in name_crops:
            try:
                card_index = crop.get("card_index")
                slot_index = crop.get("slot_index")
                image_base64 = crop.get("image")
                
                if not image_base64:
                    logger.error(f"Card {card_index} Slot {slot_index}: No image provided")
                    results.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "name": "",
                        "confidence": "LOW",
                        "error": "No image provided"
                    })
                    continue
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(image_base64)
                
                # Extract name
                extraction = self.extract_name(image_bytes)
                
                results.append({
                    "card_index": card_index,
                    "slot_index": slot_index,
                    "name": extraction["name"],
                    "confidence": extraction["confidence"]
                })
            
            except Exception as e:
                logger.error(f"Error extracting name for card {crop.get('card_index')} slot {crop.get('slot_index')}: {str(e)}")
                results.append({
                    "card_index": crop.get("card_index"),
                    "slot_index": crop.get("slot_index"),
                    "name": "",
                    "confidence": "LOW",
                    "error": str(e)
                })
        
        return results
