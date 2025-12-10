"""
Haiku-based full result extractor for cost/speed comparison.
Uses Claude 3.5 Haiku instead of Sonnet 4.5.
"""

import json
import logging
import time
import base64
import os
import boto3

logger = logging.getLogger(__name__)


class HaikuExtractionError(Exception):
    """Haiku extraction error."""
    pass


class HaikuExtractor:
    """Extract full match result using Claude Haiku 4.5 with thinking."""
    
    def __init__(self):
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # US inference profile
        
        # Load prompts
        prompt_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
        
        with open(os.path.join(prompt_dir, "system_prompt.txt"), "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(os.path.join(prompt_dir, "full_result_user_prompt.txt"), "r") as f:
            self.user_prompt = f.read().strip()
        
        logger.info(f"HaikuExtractor initialized with model: {self.model_id}")
    
    def _validate_screenshot(self, screenshot_base64: str) -> None:
        """Validate screenshot input."""
        if not screenshot_base64:
            raise HaikuExtractionError("Screenshot base64 is empty")
        
        if len(screenshot_base64) < 1000:
            raise HaikuExtractionError("Screenshot too small")
        
        if len(screenshot_base64) > 5000000:
            raise HaikuExtractionError("Screenshot too large (>5MB)")
    
    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from Claude response."""
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try markdown code blocks
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
        
        raise HaikuExtractionError(f"Could not parse JSON from response: {response_text[:200]}")
    
    def _validate_result_schema(self, data: dict) -> None:
        """Validate extraction result schema."""
        if not isinstance(data, dict):
            raise HaikuExtractionError("Response must be a JSON object")
        
        if "teams" not in data:
            raise HaikuExtractionError("Response must contain 'teams' key")
        
        teams = data.get("teams")
        if not isinstance(teams, list):
            raise HaikuExtractionError("'teams' must be a list")
        
        if len(teams) < 1 or len(teams) > 12:
            raise HaikuExtractionError(f"Expected 1-12 teams, got {len(teams)}")
        
        # Basic validation of team structure
        for i, team in enumerate(teams):
            if not isinstance(team, dict):
                raise HaikuExtractionError(f"Team {i} is not a dict")
            
            required_fields = ["card_index", "placement", "team_name", "total_kills", "players"]
            for field in required_fields:
                if field not in team:
                    raise HaikuExtractionError(f"Team {i} missing '{field}'")
            
            players = team.get("players")
            if not isinstance(players, list):
                raise HaikuExtractionError(f"Team {i}: 'players' must be a list")
            
            if len(players) < 1 or len(players) > 4:
                raise HaikuExtractionError(f"Team {i}: expected 1-4 players, got {len(players)}")
            
            # Validate each player
            for j, player in enumerate(players):
                if not isinstance(player, dict):
                    raise HaikuExtractionError(f"Team {i} Player {j} is not a dict")
                
                required_player_fields = ["slot_index", "player_name", "kills"]
                for field in required_player_fields:
                    if field not in player:
                        raise HaikuExtractionError(f"Team {i} Player {j} missing '{field}'")
                
                # Validate slot_index
                if not isinstance(player["slot_index"], int) or player["slot_index"] < 1 or player["slot_index"] > 4:
                    raise HaikuExtractionError(f"Team {i} Player {j}: invalid slot_index {player['slot_index']}")
                
                # Validate player_name
                if not isinstance(player["player_name"], str):
                    raise HaikuExtractionError(f"Team {i} Player {j}: player_name must be string")
                
                # Validate kills
                if player["kills"] is not None:
                    if not isinstance(player["kills"], int) or player["kills"] < 0 or player["kills"] > 20:
                        raise HaikuExtractionError(f"Team {i} Player {j}: invalid kills {player['kills']}")
    
    def _refine_response(self, data: dict) -> tuple:
        """
        Refine the extraction response by fixing inconsistencies.
        
        Returns:
            tuple: (refined_data, list_of_refinements)
        """
        refinements = []
        
        for team in data.get("teams", []):
            card_idx = team.get("card_index", "?")
            team_name = team.get("team_name", "Unknown")
            
            # Calculate sum of player kills
            players = team.get("players", [])
            player_kills_sum = 0
            valid_kills = True
            
            for player in players:
                kills = player.get("kills")
                if kills is not None and isinstance(kills, int):
                    player_kills_sum += kills
                else:
                    valid_kills = False
            
            # Check if total_kills matches sum
            total_kills = team.get("total_kills")
            
            if valid_kills and total_kills is not None:
                if total_kills != player_kills_sum:
                    refinements.append({
                        "card": card_idx,
                        "team": team_name,
                        "issue": "total_kills_mismatch",
                        "original": total_kills,
                        "corrected": player_kills_sum,
                        "message": f"Card {card_idx} ({team_name}): total_kills {total_kills} → {player_kills_sum}"
                    })
                    # Fix the total to match sum (individual kills are more reliable)
                    team["total_kills"] = player_kills_sum
        
        return data, refinements
    
    def extract_full_result(self, screenshot_base64: str) -> dict:
        """
        Extract complete match result from full screenshot using Haiku.
        
        Args:
            screenshot_base64: Base64-encoded full screenshot
            
        Returns:
            Dict with teams, players, kills, placements
            
        Raises:
            HaikuExtractionError: If extraction fails
        """
        logger.info(f"Incoming Haiku extraction request. Screenshot size: {len(screenshot_base64)} chars")
        
        # Validate input
        self._validate_screenshot(screenshot_base64)
        
        try:
            # Call Claude Haiku 4.5 with thinking enabled
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
                additionalModelRequestFields={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10000  # Haiku uses less thinking tokens
                    }
                },
                inferenceConfig={
                    "maxTokens": 20000,  # Haiku needs less tokens
                    "temperature": 1,  # Required for thinking
                }
            )
            latency = time.time() - start_time
            
            # Extract token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            total_tokens = input_tokens + output_tokens
            
            logger.info(
                f"Haiku call completed. Latency: {latency:.2f}s | "
                f"Tokens: {input_tokens} in, {output_tokens} out, {total_tokens} total"
            )
            
            # Extract response
            if "output" not in response or "message" not in response["output"]:
                raise HaikuExtractionError("Invalid Bedrock response structure")
            
            content_blocks = response["output"]["message"]["content"]
            if not isinstance(content_blocks, list) or len(content_blocks) == 0:
                raise HaikuExtractionError("No content in Claude response")
            
            response_text = None
            for block in content_blocks:
                if "text" in block:
                    response_text = block["text"]
                    break
            
            if not response_text:
                raise HaikuExtractionError("No text content in Claude response")
            
            logger.debug(f"Claude response: {response_text[:500]}...")
            
            # Parse JSON
            try:
                data = self._parse_json_response(response_text)
                logger.debug("JSON parsed successfully")
            except HaikuExtractionError as e:
                logger.error(f"JSON parse failed: {str(e)}")
                raise
            
            # Validate schema
            try:
                self._validate_result_schema(data)
                logger.info("Result schema validated successfully")
            except HaikuExtractionError as e:
                logger.error(f"Schema validation failed: {str(e)}")
                raise
            
            # Refine response - fix totals and report issues
            refined_data, refinements = self._refine_response(data)
            if refinements:
                logger.info(f"Applied {len(refinements)} refinements: {[r['message'] for r in refinements]}")
            
            logger.info(f"Haiku extraction successful. Teams: {len(refined_data['teams'])}")
            
            return {
                "success": True,
                "data": refined_data,
                "refinements": refinements,
                "model": "haiku-3.5",
                "latency_seconds": latency,
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
        
        except HaikuExtractionError as e:
            logger.error(f"Extraction error: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            raise HaikuExtractionError(f"Unexpected error: {str(e)}")
