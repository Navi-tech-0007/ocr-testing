"""
Two-pass extractor for improved card isolation.
Pass 1: Detect card structure (placement numbers + player counts)
Pass 2: Extract full data using the structure from Pass 1
"""

import json
import logging
import time
import base64
import os
import boto3

logger = logging.getLogger(__name__)


class TwoPassExtractionError(Exception):
    """Two-pass extraction error."""
    pass


class TwoPassExtractor:
    """Extract full match result using a two-pass approach for better card isolation."""
    
    def __init__(self):
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")
        self.haiku_model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Pass 1: Fast structure detection
        self.sonnet_model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Pass 2: Detailed extraction
        
        # Load prompts
        prompt_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
        
        with open(os.path.join(prompt_dir, "system_prompt.txt"), "r") as f:
            self.system_prompt = f.read().strip()
        
        # Single unified prompt for both passes
        with open(os.path.join(prompt_dir, "full_result_user_prompt.txt"), "r") as f:
            self.unified_prompt = f.read().strip()
        
        logger.info(f"TwoPassExtractor initialized. Pass1: Haiku, Pass2: Sonnet")
    
    def _call_claude(self, screenshot_bytes: bytes, user_prompt: str, model_id: str, use_thinking: bool = False) -> tuple:
        """Call Claude Vision with the given prompt."""
        
        config = {
            "maxTokens": 8000,
            "temperature": 1 if use_thinking else 0,
        }
        
        additional_fields = {}
        if use_thinking:
            additional_fields["thinking"] = {
                "type": "enabled",
                "budget_tokens": 5000
            }
        
        kwargs = {
            "modelId": model_id,
            "system": [{"text": self.system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": screenshot_bytes},
                            },
                        },
                        {"text": user_prompt}
                    ],
                }
            ],
            "inferenceConfig": config,
        }
        
        if additional_fields:
            kwargs["additionalModelRequestFields"] = additional_fields
        
        response = self.bedrock_client.converse(**kwargs)
        
        # Extract response text
        content_blocks = response["output"]["message"]["content"]
        response_text = None
        for block in content_blocks:
            if "text" in block:
                response_text = block["text"]
                break
        
        # Extract token usage
        usage = response.get("usage", {})
        tokens = {
            "input": usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
        }
        
        return response_text, tokens
    
    def _parse_json(self, text: str) -> dict:
        """Parse JSON from response text."""
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to extract JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        
        raise TwoPassExtractionError(f"Could not parse JSON: {text[:200]}")
    
    def _build_pass2_prompt(self, structure: dict) -> str:
        """Build Pass 2 prompt with card structure constraints."""
        
        cards = structure.get("cards", [])
        
        # Build constraint text
        constraint_lines = [
            "\n\n**CARD STRUCTURE CONSTRAINTS (FROM PASS 1 - FOLLOW EXACTLY):**",
            "The following cards are visible with their player counts:",
        ]
        
        for card in cards:
            placement = card.get("placement")
            count = card.get("visible_players")
            constraint_lines.append(f"- Placement {placement}: {count} players visible")
        
        constraint_lines.extend([
            "",
            "YOU MUST extract EXACTLY these player counts for each card.",
            "Do NOT add or remove players from what is specified above.",
            "If a card shows 2 players, return ONLY 2 players for that card.",
        ])
        
        constraint_text = "\n".join(constraint_lines)
        
        # Append constraints to unified prompt
        return self.unified_prompt + constraint_text
    
    def _refine_response(self, data: dict) -> tuple:
        """Refine the extraction response by fixing inconsistencies."""
        refinements = []
        
        for team in data.get("teams", []):
            card_idx = team.get("card_index", "?")
            team_name = team.get("team_name", "Unknown")
            
            players = team.get("players", [])
            player_kills_sum = sum(p.get("kills", 0) or 0 for p in players if p.get("kills") is not None)
            
            total_kills = team.get("total_kills")
            if total_kills is not None and total_kills != player_kills_sum:
                refinements.append({
                    "card": card_idx,
                    "team": team_name,
                    "issue": "total_kills_mismatch",
                    "original": total_kills,
                    "corrected": player_kills_sum,
                    "message": f"Card {card_idx} ({team_name}): total_kills {total_kills} → {player_kills_sum}"
                })
                team["total_kills"] = player_kills_sum
        
        return data, refinements
    
    def extract_full_result(self, screenshot_base64: str) -> dict:
        """
        Extract complete match result using two-pass approach.
        
        Pass 1: Detect card structure (placement numbers + player counts)
        Pass 2: Extract full data using the structure constraints
        """
        logger.info(f"Starting two-pass extraction. Screenshot size: {len(screenshot_base64)} chars")
        
        try:
            screenshot_bytes = base64.b64decode(screenshot_base64)
        except Exception as e:
            raise TwoPassExtractionError(f"Invalid base64: {str(e)}")
        
        total_tokens = {"input": 0, "output": 0}
        
        try:
            # PASS 1: Detect card structure (using Haiku for speed)
            logger.info("Pass 1: Detecting card structure with Haiku...")
            start_time = time.time()
            
            pass1_response, pass1_tokens = self._call_claude(
                screenshot_bytes, 
                self.unified_prompt,
                model_id=self.haiku_model_id,
                use_thinking=False  # Quick pass, no thinking needed
            )
            
            pass1_latency = time.time() - start_time
            total_tokens["input"] += pass1_tokens["input"]
            total_tokens["output"] += pass1_tokens["output"]
            
            logger.info(f"Pass 1 complete. Latency: {pass1_latency:.2f}s | Tokens: {pass1_tokens}")
            
            # Parse structure
            structure = self._parse_json(pass1_response)
            cards = structure.get("cards", [])
            logger.info(f"Pass 1 detected {len(cards)} cards: {cards}")
            
            # PASS 2: Extract full data with structure constraints (using Sonnet)
            logger.info("Pass 2: Extracting full data with Sonnet...")
            pass2_start = time.time()
            
            pass2_prompt = self._build_pass2_prompt(structure)
            
            pass2_response, pass2_tokens = self._call_claude(
                screenshot_bytes,
                pass2_prompt,
                model_id=self.haiku_model_id,
                use_thinking=True  # Use thinking for accurate extraction
            )
            
            pass2_latency = time.time() - pass2_start
            total_tokens["input"] += pass2_tokens["input"]
            total_tokens["output"] += pass2_tokens["output"]
            
            logger.info(f"Pass 2 complete. Latency: {pass2_latency:.2f}s | Tokens: {pass2_tokens}")
            
            # Parse and validate
            data = self._parse_json(pass2_response)
            
            # Validate structure
            if "teams" not in data:
                raise TwoPassExtractionError("No 'teams' in response")
            
            # Refine response
            refined_data, refinements = self._refine_response(data)
            
            total_latency = pass1_latency + pass2_latency
            
            logger.info(f"Two-pass extraction complete. Total latency: {total_latency:.2f}s | Teams: {len(refined_data['teams'])}")
            
            return {
                "success": True,
                "data": refined_data,
                "refinements": refinements,
                "model": "haiku+sonnet-twopass",
                "pass1_structure": structure,
                "pass1_latency": pass1_latency,
                "pass2_latency": pass2_latency,
                "latency_seconds": total_latency,
                "tokens": {
                    "input_tokens": total_tokens["input"],
                    "output_tokens": total_tokens["output"],
                    "total_tokens": total_tokens["input"] + total_tokens["output"]
                }
            }
        
        except TwoPassExtractionError:
            raise
        except Exception as e:
            logger.exception(f"Two-pass extraction error: {str(e)}")
            raise TwoPassExtractionError(f"Unexpected error: {str(e)}")
