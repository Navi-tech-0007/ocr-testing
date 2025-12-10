"""
Full-screenshot Claude Vision extraction service.

Extracts entire match result (all teams, players, kills) from a single screenshot.
Used for organizer verification and final result validation.
"""

import json
import base64
import logging
import time
from pathlib import Path
from typing import Optional
import boto3

logger = logging.getLogger(__name__)


class FullResultExtractionError(Exception):
    """Custom exception for full result extraction errors."""
    pass


class FullResultExtractor:
    """Service for extracting complete match results using Claude Vision."""
    
    def __init__(self):
        """Initialize the extractor with Claude client."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        logger.info("FullResultExtractor initialized")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load Claude prompts from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "full_result_system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "full_result_user_prompt.txt", "r") as f:
            self.user_prompt = f.read().strip()
    
    def _validate_screenshot(self, screenshot_base64: str) -> None:
        """
        Validate screenshot base64 format.
        
        Args:
            screenshot_base64: Base64-encoded screenshot
            
        Raises:
            FullResultExtractionError: If invalid
        """
        if not screenshot_base64:
            raise FullResultExtractionError("Screenshot base64 is empty")
        
        if not isinstance(screenshot_base64, str):
            raise FullResultExtractionError("Screenshot must be base64 string")
        
        # Try to decode to verify it's valid base64
        try:
            decoded = base64.b64decode(screenshot_base64)
            if len(decoded) < 1000:  # Minimum reasonable screenshot size
                raise FullResultExtractionError("Screenshot too small (< 1KB)")
        except Exception as e:
            raise FullResultExtractionError(f"Invalid base64: {str(e)}")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from Claude.
        
        Handles cases where Claude wraps JSON in markdown or extra text.
        
        Args:
            response_text: Raw response text from Claude
            
        Returns:
            Parsed JSON dict
            
        Raises:
            FullResultExtractionError: If JSON is invalid
        """
        response_text = response_text.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                try:
                    return json.loads(response_text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to extract JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(response_text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        
        raise FullResultExtractionError(f"Could not parse JSON from response: {response_text[:200]}")
    
    def _validate_result_schema(self, data: dict) -> None:
        """
        Validate the extracted result schema.
        
        Args:
            data: Parsed JSON data
            
        Raises:
            FullResultExtractionError: If schema is invalid
        """
        if not isinstance(data, dict):
            raise FullResultExtractionError("Response must be a JSON object")
        
        if "teams" not in data:
            raise FullResultExtractionError("Response must contain 'teams' key")
        
        teams = data.get("teams")
        if not isinstance(teams, list):
            raise FullResultExtractionError("'teams' must be a list")
        
        if len(teams) < 1 or len(teams) > 12:
            raise FullResultExtractionError(f"Expected 1-12 teams, got {len(teams)}")
        
        # Validate each team
        for i, team in enumerate(teams):
            if not isinstance(team, dict):
                raise FullResultExtractionError(f"Team {i} is not a dict")
            
            # Check required fields
            required_fields = ["card_index", "placement", "team_name", "total_kills", "players"]
            for field in required_fields:
                if field not in team:
                    raise FullResultExtractionError(f"Team {i} missing '{field}'")
            
            # Validate card_index
            if not isinstance(team["card_index"], int) or team["card_index"] < 1 or team["card_index"] > 12:
                raise FullResultExtractionError(f"Team {i}: invalid card_index {team['card_index']}")
            
            # Validate placement
            if not isinstance(team["placement"], int) or team["placement"] < 1 or team["placement"] > 12:
                raise FullResultExtractionError(f"Team {i}: invalid placement {team['placement']}")
            
            # Validate team_name
            if not isinstance(team["team_name"], str):
                raise FullResultExtractionError(f"Team {i}: team_name must be string")
            
            # Validate total_kills
            if team["total_kills"] is not None:
                if not isinstance(team["total_kills"], int) or team["total_kills"] < 0:
                    raise FullResultExtractionError(f"Team {i}: invalid total_kills {team['total_kills']}")
            
            # Validate players
            players = team.get("players")
            if not isinstance(players, list):
                raise FullResultExtractionError(f"Team {i}: 'players' must be a list")
            
            if len(players) != 4:
                raise FullResultExtractionError(f"Team {i}: expected 4 players, got {len(players)}")
            
            for j, player in enumerate(players):
                if not isinstance(player, dict):
                    raise FullResultExtractionError(f"Team {i} Player {j}: not a dict")
                
                # Check required fields
                player_fields = ["slot_index", "player_name", "kills"]
                for field in player_fields:
                    if field not in player:
                        raise FullResultExtractionError(f"Team {i} Player {j}: missing '{field}'")
                
                # Validate slot_index
                if not isinstance(player["slot_index"], int) or player["slot_index"] < 1 or player["slot_index"] > 4:
                    raise FullResultExtractionError(f"Team {i} Player {j}: invalid slot_index {player['slot_index']}")
                
                # Validate player_name
                if not isinstance(player["player_name"], str):
                    raise FullResultExtractionError(f"Team {i} Player {j}: player_name must be string")
                
                # Validate kills
                if player["kills"] is not None:
                    if not isinstance(player["kills"], int) or player["kills"] < 0 or player["kills"] > 20:
                        raise FullResultExtractionError(f"Team {i} Player {j}: invalid kills {player['kills']}")
    
    def extract_full_result(self, screenshot_base64: str) -> dict:
        """
        Extract complete match result from full screenshot.
        
        Args:
            screenshot_base64: Base64-encoded full screenshot
            
        Returns:
            Dict with teams, players, kills, placements
            
        Raises:
            FullResultExtractionError: If extraction fails
        """
        logger.info(f"Incoming full result extraction request. Screenshot size: {len(screenshot_base64)} chars")
        
        # Validate input
        self._validate_screenshot(screenshot_base64)
        
        try:
            # Call Claude Vision
            start_time = time.time()
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
                                        "bytes": base64.b64decode(screenshot_base64),
                                    },
                                },
                            },
                            {
                                "text": self.user_prompt,
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 4000,
                    "temperature": 0,
                    "topP": 1,
                }
            )
            latency = time.time() - start_time
            
            # Extract token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            total_tokens = input_tokens + output_tokens
            
            logger.info(
                f"Claude Vision call completed. Latency: {latency:.2f}s | "
                f"Tokens: {input_tokens} in, {output_tokens} out, {total_tokens} total"
            )
            
            # Extract response
            if "output" not in response or "message" not in response["output"]:
                raise FullResultExtractionError("Invalid Bedrock response structure")
            
            content_blocks = response["output"]["message"]["content"]
            if not isinstance(content_blocks, list) or len(content_blocks) == 0:
                raise FullResultExtractionError("No content in Claude response")
            
            response_text = None
            for block in content_blocks:
                if "text" in block:
                    response_text = block["text"]
                    break
            
            if not response_text:
                raise FullResultExtractionError("No text content in Claude response")
            
            logger.debug(f"Claude response: {response_text[:500]}...")
            
            # Parse JSON
            try:
                data = self._parse_json_response(response_text)
                logger.debug("JSON parsed successfully")
            except FullResultExtractionError as e:
                logger.error(f"JSON parse failed: {str(e)}")
                raise
            
            # Validate schema
            try:
                self._validate_result_schema(data)
                logger.info("Result schema validated successfully")
            except FullResultExtractionError as e:
                logger.error(f"Schema validation failed: {str(e)}")
                raise
            
            logger.info(f"Full result extraction successful. Teams: {len(data['teams'])}")
            
            return {
                "success": True,
                "data": data,
                "latency_seconds": latency,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens
                }
            }
        
        except FullResultExtractionError as e:
            logger.error(f"Extraction error: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            raise FullResultExtractionError(f"Unexpected error: {str(e)}")
