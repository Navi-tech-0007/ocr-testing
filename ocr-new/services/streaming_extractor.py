"""
Streaming extractor for real-time Claude thinking and response.

Uses Bedrock's converse_stream API to stream thinking blocks and final response.
"""

import json
import base64
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
import boto3

logger = logging.getLogger(__name__)


class StreamingExtractionError(Exception):
    """Custom exception for streaming extraction errors."""
    pass


class StreamingExtractor:
    """Streams Claude thinking and extraction in real-time."""
    
    def __init__(self):
        """Initialize the streaming extractor."""
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        logger.info(f"StreamingExtractor initialized with model: {self.model_id}")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load Claude prompts from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        with open(prompts_dir / "full_result_system_prompt.txt", "r") as f:
            self.system_prompt = f.read().strip()
        
        with open(prompts_dir / "full_result_user_prompt.txt", "r") as f:
            self.user_prompt = f.read().strip()
    
    async def stream_extraction(self, screenshot_base64: str) -> AsyncGenerator[dict, None]:
        """
        Stream Claude thinking and extraction result.
        
        Yields events in real-time:
        - thinking: Claude's internal reasoning
        - text: Final JSON response
        - error: Any errors during processing
        - done: Extraction complete
        
        Args:
            screenshot_base64: Base64-encoded screenshot
            
        Yields:
            Dict events with type and content
        """
        if not screenshot_base64:
            yield {
                "type": "error",
                "error": "Screenshot base64 is empty"
            }
            return
        
        try:
            # Validate screenshot
            try:
                decoded = base64.b64decode(screenshot_base64)
                if len(decoded) < 1000:
                    yield {
                        "type": "error",
                        "error": "Screenshot too small (< 1KB)"
                    }
                    return
            except Exception as e:
                yield {
                    "type": "error",
                    "error": f"Invalid base64: {str(e)}"
                }
                return
            
            logger.info("Starting streaming extraction")
            yield {
                "type": "status",
                "message": "Sending image to Claude..."
            }
            
            # Configure thinking for extended reasoning
            reasoning_config = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
            
            # Call Bedrock with streaming (thinking enabled)
            start_time = time.time()
            response_stream = self.bedrock_client.converse_stream(
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
                additionalModelRequestFields=reasoning_config,
                inferenceConfig={
                    "maxTokens": 12000,
                    "temperature": 1,  # Required for thinking
                }
            )
            
            # Process stream events
            full_response = ""
            thinking_text = ""
            thinking_buffer = ""  # Buffer to accumulate thinking chunks
            text_buffer = ""  # Buffer to accumulate text chunks
            BUFFER_SIZE = 100  # Send every 100 chars for smoother display
            
            thinking_started = False
            
            for event in response_stream["stream"]:
                # Log all event keys to understand structure
                event_keys = list(event.keys())
                logger.info(f"Stream event: {event_keys}")
                
                # Handle thinking blocks - check for reasoningContent
                if "contentBlockStart" in event:
                    block_start = event["contentBlockStart"]
                    logger.info(f"ContentBlockStart: {block_start}")
                    content_block = block_start.get("contentBlock", {})
                    # Check for reasoningContent (Bedrock format) or thinking
                    if content_block.get("reasoningContent") or content_block.get("thinking"):
                        thinking_started = True
                        logger.info("Thinking/Reasoning block started")
                        yield {
                            "type": "thinking_start",
                            "message": "Claude is thinking..."
                        }
                
                # Handle thinking delta - check multiple formats
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    logger.info(f"Delta keys: {delta.keys()}")
                    
                    # Check for reasoningContent (Bedrock) or thinkingDelta
                    reasoning = delta.get("reasoningContent") or delta.get("thinkingDelta")
                    if reasoning:
                        thinking_chunk = reasoning.get("text") or reasoning.get("thinking") or str(reasoning)
                        if thinking_chunk:
                            thinking_text += thinking_chunk
                            thinking_buffer += thinking_chunk
                            # Send when buffer is large enough
                            if len(thinking_buffer) >= BUFFER_SIZE:
                                logger.info(f"Sending thinking ({len(thinking_buffer)} chars)")
                                yield {
                                    "type": "thinking",
                                    "content": thinking_buffer
                                }
                                thinking_buffer = ""
                    elif "text" in delta:
                        text_chunk = delta["text"]
                        full_response += text_chunk
                        text_buffer += text_chunk
                        if len(text_buffer) >= BUFFER_SIZE:
                            yield {
                                "type": "text",
                                "content": text_buffer
                            }
                            text_buffer = ""
                    elif "textDelta" in delta:
                        text_chunk = delta["textDelta"]["text"]
                        full_response += text_chunk
                        text_buffer += text_chunk
                        if len(text_buffer) >= BUFFER_SIZE:
                            yield {
                                "type": "text",
                                "content": text_buffer
                            }
                            text_buffer = ""
                
                # Handle message stop
                if "messageStop" in event:
                    # Flush remaining buffers
                    if thinking_buffer:
                        yield {
                            "type": "thinking",
                            "content": thinking_buffer
                        }
                    if text_buffer:
                        yield {
                            "type": "text",
                            "content": text_buffer
                        }
                    
                    latency = time.time() - start_time
                    yield {
                        "type": "status",
                        "message": f"Extraction complete in {latency:.2f}s"
                    }
                
                # Handle metadata
                if "metadata" in event:
                    metadata = event["metadata"]
                    if "usage" in metadata:
                        usage = metadata["usage"]
                        yield {
                            "type": "tokens",
                            "input": usage.get("inputTokens", 0),
                            "output": usage.get("outputTokens", 0),
                            "total": usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
                        }
            
            # Parse and validate final response
            if full_response:
                try:
                    result = self._parse_json_response(full_response)
                    self._validate_result_schema(result)
                    
                    # Refine response - fix totals and report issues
                    refined_result, refinements = self._refine_response(result)
                    
                    if refinements:
                        yield {
                            "type": "refinements",
                            "fixes": refinements
                        }
                    
                    yield {
                        "type": "result",
                        "data": refined_result
                    }
                except Exception as e:
                    yield {
                        "type": "error",
                        "error": f"JSON parse/validation error: {str(e)}"
                    }
            
            yield {
                "type": "done"
            }
        
        except Exception as e:
            logger.exception(f"Streaming extraction error: {str(e)}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
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
        
        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
    
    def _validate_result_schema(self, data: dict) -> None:
        """Validate extraction result schema."""
        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")
        
        if "teams" not in data:
            raise ValueError("Response must contain 'teams' key")
        
        teams = data.get("teams")
        if not isinstance(teams, list):
            raise ValueError("'teams' must be a list")
        
        if len(teams) < 1 or len(teams) > 12:
            raise ValueError(f"Expected 1-12 teams, got {len(teams)}")
        
        # Basic validation of team structure
        for i, team in enumerate(teams):
            if not isinstance(team, dict):
                raise ValueError(f"Team {i} is not a dict")
            
            required_fields = ["card_index", "placement", "team_name", "total_kills", "players"]
            for field in required_fields:
                if field not in team:
                    raise ValueError(f"Team {i} missing '{field}'")
            
            players = team.get("players")
            if not isinstance(players, list):
                raise ValueError(f"Team {i}: 'players' must be a list")
            
            if len(players) < 1 or len(players) > 4:
                raise ValueError(f"Team {i}: expected 1-4 players, got {len(players)}")
    
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
                        "message": f"Card {card_idx} ({team_name}): total_kills {total_kills} → {player_kills_sum} (sum of player kills)"
                    })
                    # Fix the total to match sum (individual kills are more reliable)
                    team["total_kills"] = player_kills_sum
            
            # Validate player kill counts are reasonable (0-20)
            for player in players:
                kills = player.get("kills")
                if kills is not None and isinstance(kills, int):
                    if kills < 0 or kills > 20:
                        refinements.append({
                            "card": card_idx,
                            "team": team_name,
                            "player": player.get("player_name"),
                            "issue": "suspicious_kills",
                            "value": kills,
                            "message": f"Card {card_idx}: {player.get('player_name')} has {kills} kills (unusual)"
                        })
        
        return data, refinements
