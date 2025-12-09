import json
import base64
import os
import logging
import time
from typing import Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
import boto3

logger = logging.getLogger(__name__)

class CardDetectionError(Exception):
    """Custom exception for card detection errors."""
    pass


class TeamCardDetector:
    """Service for detecting team card bounding boxes from Free Fire scoreboard screenshots."""
    
    def __init__(self):
        """Initialize the detector with Bedrock client and prompts."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.enabled = os.getenv("OCR_TEAM_CARD_DETECTION_ENABLED", "true").lower() == "true"
        logger.info(f"TeamCardDetector initialized. Enabled: {self.enabled}")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load system, user, and retry prompts from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "user_prompt.txt", "r") as f:
            self.user_prompt = f.read().strip()
        
        with open(prompts_dir / "retry_prompt.txt", "r") as f:
            self.retry_prompt = f.read().strip()
    
    def _validate_image(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Validate image and return dimensions.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            CardDetectionError: If image is invalid
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            if width <= 0 or height <= 0:
                raise CardDetectionError("Invalid image dimensions")
            return width, height
        except Exception as e:
            raise CardDetectionError(f"Invalid image: {str(e)}")
    
    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.standard_b64encode(image_bytes).decode("utf-8")
    
    def _call_claude_vision(self, image_base64: str, prompt_text: str, is_retry: bool = False) -> str:
        """
        Call Claude 3.7 Sonnet Vision via AWS Bedrock with image and prompt.
        Uses extended thinking for better accuracy.
        
        Args:
            image_base64: Base64-encoded image
            prompt_text: The user prompt to send
            is_retry: Whether this is a retry attempt
            
        Returns:
            Raw response text from Claude
        """
        # Configure thinking for better accuracy
        reasoning_config = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 5000
            }
        }
        
        # Call Bedrock using converse API with thinking enabled
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
            additionalModelRequestFields=reasoning_config,
            inferenceConfig={
                "maxTokens": 6000,
                "temperature": 1,
                "topP": 1,
            }
        )
        
        # Extract text from response (skip reasoning blocks)
        if "output" in response and "message" in response["output"]:
            content_blocks = response["output"]["message"]["content"]
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if "text" in block:
                        return block["text"]
        
        raise CardDetectionError("No text content in Bedrock response")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from Claude.
        
        Args:
            response_text: Raw response text from Claude
            
        Returns:
            Parsed JSON dict
            
        Raises:
            CardDetectionError: If JSON is invalid
        """
        # Try to extract JSON from response (in case there's extra text)
        response_text = response_text.strip()
        
        # Try direct parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(response_text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        
        raise CardDetectionError("Invalid JSON response from Claude")
    
    def _validate_cards_schema(self, data: dict, image_width: int, image_height: int) -> list[dict]:
        """
        Validate cards schema and bounds.
        
        Args:
            data: Parsed JSON data
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Validated and cleaned cards list
            
        Raises:
            CardDetectionError: If schema is invalid
        """
        if not isinstance(data, dict):
            raise CardDetectionError("Response must be a JSON object")
        
        if "cards" not in data:
            raise CardDetectionError("Response must contain 'cards' key")
        
        cards = data["cards"]
        if not isinstance(cards, list):
            raise CardDetectionError("'cards' must be a list")
        
        validated_cards = []
        
        for i, card in enumerate(cards):
            if not isinstance(card, dict):
                raise CardDetectionError(f"Card {i} is not a dict")
            
            # Validate required fields
            if "card_index" not in card:
                raise CardDetectionError(f"Card {i} missing 'card_index'")
            if "bounds" not in card:
                raise CardDetectionError(f"Card {i} missing 'bounds'")
            
            card_index = card.get("card_index")
            bounds = card.get("bounds")
            
            # Validate card_index is integer
            if not isinstance(card_index, int):
                raise CardDetectionError(f"Card {i} 'card_index' must be integer")
            
            # Validate bounds structure
            if not isinstance(bounds, dict):
                raise CardDetectionError(f"Card {i} 'bounds' must be dict")
            
            for coord in ["x1", "y1", "x2", "y2"]:
                if coord not in bounds:
                    raise CardDetectionError(f"Card {i} bounds missing '{coord}'")
                if not isinstance(bounds[coord], int):
                    raise CardDetectionError(f"Card {i} bounds '{coord}' must be integer")
            
            x1, y1, x2, y2 = bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]
            
            # Validate bounds logic
            if x1 >= x2:
                raise CardDetectionError(f"Card {i} invalid bounds: x1 >= x2")
            if y1 >= y2:
                raise CardDetectionError(f"Card {i} invalid bounds: y1 >= y2")
            
            # Validate bounds within image
            if not (0 <= x1 <= image_width):
                raise CardDetectionError(f"Card {i} x1={x1} out of bounds [0, {image_width}]")
            if not (0 <= y1 <= image_height):
                raise CardDetectionError(f"Card {i} y1={y1} out of bounds [0, {image_height}]")
            if not (0 <= x2 <= image_width):
                raise CardDetectionError(f"Card {i} x2={x2} out of bounds [0, {image_width}]")
            if not (0 <= y2 <= image_height):
                raise CardDetectionError(f"Card {i} y2={y2} out of bounds [0, {image_height}]")
            
            # Add cleaned card (only card_index and bounds)
            validated_cards.append({
                "card_index": card_index,
                "bounds": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            })
        
        return validated_cards
    
    def detect_cards(self, image_bytes: bytes) -> dict:
        """
        Detect team card bounding boxes from scoreboard screenshot.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dict with cards list and optional error field
            
        Raises:
            CardDetectionError: If detection fails
        """
        if not self.enabled:
            raise CardDetectionError("Team card detection is disabled (OCR_TEAM_CARD_DETECTION_ENABLED=false)")
        
        # Log incoming request
        logger.info(f"Incoming detection request. Image size: {len(image_bytes)} bytes")
        
        # Validate image
        image_width, image_height = self._validate_image(image_bytes)
        logger.info(f"Image validated. Dimensions: {image_width}x{image_height}")
        
        # Convert to base64
        image_base64 = self._image_to_base64(image_bytes)
        
        # Call Claude Vision (with retry)
        start_time = time.time()
        response_text = self._call_claude_vision(image_base64, self.user_prompt, is_retry=False)
        latency = time.time() - start_time
        logger.info(f"Bedrock Claude call completed. Latency: {latency:.2f}s")
        
        try:
            data = self._parse_json_response(response_text)
            logger.debug("JSON parsed successfully on first attempt")
        except CardDetectionError as e:
            logger.warning(f"JSON parse failed on first attempt: {str(e)}. Retrying...")
            # Retry with retry prompt
            start_time = time.time()
            response_text = self._call_claude_vision(image_base64, self.retry_prompt, is_retry=True)
            latency = time.time() - start_time
            logger.info(f"Bedrock Claude retry call completed. Latency: {latency:.2f}s")
            try:
                data = self._parse_json_response(response_text)
                logger.info("JSON parsed successfully on retry")
            except CardDetectionError as e:
                logger.error(f"JSON parse failed after retry: {str(e)}")
                raise CardDetectionError(f"Failed to parse JSON after retry: {str(e)}")
        
        # Validate schema
        cards = self._validate_cards_schema(data, image_width, image_height)
        logger.info(f"Schema validated. Cards found: {len(cards)}")
        
        # Sort by card_index
        cards.sort(key=lambda c: c["card_index"])
        
        # Handle no cards case
        if not cards:
            logger.warning("No cards detected in image")
            return {
                "cards": [],
                "error": "NO_CARDS_DETECTED"
            }
        
        logger.info(f"Detection successful. Returning {len(cards)} card(s)")
        return {
            "cards": cards
        }
