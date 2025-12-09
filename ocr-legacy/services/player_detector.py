import json
import base64
import logging
import time
from typing import Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
import boto3

logger = logging.getLogger(__name__)

class PlayerDetectionError(Exception):
    """Custom exception for player detection errors."""
    pass


class PlayerSlotDetector:
    """Service for detecting player slot bounding boxes from team card images."""
    
    def __init__(self):
        """Initialize the detector with Bedrock client and prompts."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.enabled = True
        logger.info("PlayerSlotDetector initialized")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load system, user, and retry prompts from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "step2_system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "step2_user_prompt.txt", "r") as f:
            self.user_prompt = f.read().strip()
        
        with open(prompts_dir / "step2_retry_prompt.txt", "r") as f:
            self.retry_prompt = f.read().strip()
    
    def _validate_image(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Validate image and return dimensions.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            PlayerDetectionError: If image is invalid
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            if width <= 0 or height <= 0:
                raise PlayerDetectionError("Invalid image dimensions")
            return width, height
        except Exception as e:
            raise PlayerDetectionError(f"Invalid image: {str(e)}")
    
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
        reasoning_config = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 3000
            }
        }
        
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
        
        if "output" in response and "message" in response["output"]:
            content_blocks = response["output"]["message"]["content"]
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if "text" in block:
                        return block["text"]
        
        raise PlayerDetectionError("No text content in Bedrock response")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from Claude.
        
        Args:
            response_text: Raw response text from Claude
            
        Returns:
            Parsed JSON dict
            
        Raises:
            PlayerDetectionError: If JSON is invalid
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
        
        raise PlayerDetectionError("Invalid JSON response from Claude")
    
    def _validate_players_schema(self, data: dict, image_width: int, image_height: int) -> list[dict]:
        """
        Validate players schema and bounding boxes.
        
        Args:
            data: Parsed JSON data
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Validated players list
            
        Raises:
            PlayerDetectionError: If schema is invalid
        """
        if not isinstance(data, dict):
            raise PlayerDetectionError("Response must be a JSON object")
        
        if "players" not in data:
            raise PlayerDetectionError("Response must contain 'players' key")
        
        players = data["players"]
        if not isinstance(players, list):
            raise PlayerDetectionError("'players' must be a list")
        
        if len(players) != 4:
            raise PlayerDetectionError(f"Expected exactly 4 players, got {len(players)}")
        
        validated_players = []
        expected_positions = ["ROW1_LEFT", "ROW1_RIGHT", "ROW2_LEFT", "ROW2_RIGHT"]
        
        for i, player in enumerate(players):
            if not isinstance(player, dict):
                raise PlayerDetectionError(f"Player {i} is not a dict")
            
            # Validate required fields
            if "slot_index" not in player:
                raise PlayerDetectionError(f"Player {i} missing 'slot_index'")
            if "position" not in player:
                raise PlayerDetectionError(f"Player {i} missing 'position'")
            if "name_box" not in player:
                raise PlayerDetectionError(f"Player {i} missing 'name_box'")
            if "kill_box" not in player:
                raise PlayerDetectionError(f"Player {i} missing 'kill_box'")
            
            slot_index = player.get("slot_index")
            position = player.get("position")
            name_box = player.get("name_box")
            kill_box = player.get("kill_box")
            
            # Validate slot_index
            if not isinstance(slot_index, int) or slot_index < 1 or slot_index > 4:
                raise PlayerDetectionError(f"Player {i} 'slot_index' must be 1-4, got {slot_index}")
            
            # Validate position
            if position not in expected_positions:
                raise PlayerDetectionError(f"Player {i} invalid position: {position}")
            
            # Validate bounding boxes
            for box_name, box in [("name_box", name_box), ("kill_box", kill_box)]:
                if not isinstance(box, dict):
                    raise PlayerDetectionError(f"Player {i} '{box_name}' must be dict")
                
                for coord in ["x1", "y1", "x2", "y2"]:
                    if coord not in box:
                        raise PlayerDetectionError(f"Player {i} {box_name} missing '{coord}'")
                    if not isinstance(box[coord], int):
                        raise PlayerDetectionError(f"Player {i} {box_name} '{coord}' must be integer")
                
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                
                if x1 >= x2:
                    raise PlayerDetectionError(f"Player {i} {box_name}: x1 >= x2")
                if y1 >= y2:
                    raise PlayerDetectionError(f"Player {i} {box_name}: y1 >= y2")
                
                if not (0 <= x1 <= image_width):
                    raise PlayerDetectionError(f"Player {i} {box_name} x1={x1} out of bounds")
                if not (0 <= y1 <= image_height):
                    raise PlayerDetectionError(f"Player {i} {box_name} y1={y1} out of bounds")
                if not (0 <= x2 <= image_width):
                    raise PlayerDetectionError(f"Player {i} {box_name} x2={x2} out of bounds")
                if not (0 <= y2 <= image_height):
                    raise PlayerDetectionError(f"Player {i} {box_name} y2={y2} out of bounds")
            
            validated_players.append({
                "slot_index": slot_index,
                "position": position,
                "name_box": {
                    "x1": name_box["x1"],
                    "y1": name_box["y1"],
                    "x2": name_box["x2"],
                    "y2": name_box["y2"],
                },
                "kill_box": {
                    "x1": kill_box["x1"],
                    "y1": kill_box["y1"],
                    "x2": kill_box["x2"],
                    "y2": kill_box["y2"],
                }
            })
        
        # Sort by slot_index
        validated_players.sort(key=lambda p: p["slot_index"])
        
        return validated_players
    
    def detect_players(self, image_bytes: bytes) -> dict:
        """
        Detect player slots from team card image.
        
        Args:
            image_bytes: Raw image bytes of cropped team card
            
        Returns:
            Dict with players list
            
        Raises:
            PlayerDetectionError: If detection fails
        """
        logger.info(f"Incoming player detection request. Image size: {len(image_bytes)} bytes")
        
        # Validate image
        image_width, image_height = self._validate_image(image_bytes)
        logger.info(f"Image validated. Dimensions: {image_width}x{image_height}")
        
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
        except PlayerDetectionError as e:
            logger.warning(f"JSON parse failed on first attempt: {str(e)}. Retrying...")
            start_time = time.time()
            response_text = self._call_claude_vision(image_base64, self.retry_prompt, is_retry=True)
            latency = time.time() - start_time
            logger.info(f"Bedrock Claude retry call completed. Latency: {latency:.2f}s")
            try:
                data = self._parse_json_response(response_text)
                logger.info("JSON parsed successfully on retry")
            except PlayerDetectionError as e:
                logger.error(f"JSON parse failed after retry: {str(e)}")
                raise PlayerDetectionError(f"Failed to parse JSON after retry: {str(e)}")
        
        # Validate schema
        players = self._validate_players_schema(data, image_width, image_height)
        logger.info(f"Schema validated. Players found: {len(players)}")
        
        logger.info(f"Detection successful. Returning {len(players)} player(s)")
        return {
            "players": players
        }
