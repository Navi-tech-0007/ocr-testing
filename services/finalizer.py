import logging
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# Input Models
class BoundsModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OcrCardBounds(BaseModel):
    card_index: int
    approx_visual_rank: Optional[int] = None
    bounds: BoundsModel


class OcrPlayerBoxes(BaseModel):
    card_index: int
    slot_index: int
    position: Literal["ROW1_LEFT", "ROW1_RIGHT", "ROW2_LEFT", "ROW2_RIGHT"]
    name_box: BoundsModel
    kill_box: BoundsModel


class OcrKillResult(BaseModel):
    card_index: int
    slot_index: int
    kills: Optional[int] = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    
    @validator('kills')
    def validate_kills(cls, v):
        if v is not None and (v < 0 or v > 20):
            logger.warning(f"Kill count {v} out of range, setting to None")
            return None
        return v


class OcrNameResult(BaseModel):
    card_index: int
    slot_index: int
    name: str
    confidence: Literal["HIGH", "LOW"]
    
    @validator('name')
    def validate_name(cls, v):
        # Normalize whitespace-only names to empty string
        if v and v.strip() == "":
            return ""
        return v


class OcrFinalizationRequest(BaseModel):
    match_id: Optional[str] = None
    step1_cards: list[OcrCardBounds]
    step2_players: list[OcrPlayerBoxes]
    step3_kills: list[OcrKillResult]
    step4_names: list[OcrNameResult]


# Output Models
class OcrPlayerFinal(BaseModel):
    slot_index: int
    position: str
    name: str
    name_confidence: str
    kills: Optional[int] = None
    kills_confidence: str


class OcrTeamFinal(BaseModel):
    card_index: int
    approx_visual_rank: Optional[int] = None
    players: list[OcrPlayerFinal]
    total_kills: int
    overall_confidence: str


class OcrMatchFinal(BaseModel):
    match_id: Optional[str] = None
    teams: list[OcrTeamFinal]


class OcrFinalizer:
    """Service for finalizing and assembling OCR results from all 4 steps."""
    
    def __init__(self):
        logger.info("OcrFinalizer initialized")
    
    def finalize(self, request: OcrFinalizationRequest) -> OcrMatchFinal:
        """
        Assemble and validate OCR results from all 4 steps.
        
        Args:
            request: OcrFinalizationRequest with all step outputs
            
        Returns:
            OcrMatchFinal with assembled teams and players
            
        Raises:
            ValueError: If critical data is missing
        """
        logger.info(f"Finalizing OCR results for match_id={request.match_id}")
        
        # Validate step data presence
        if not request.step1_cards:
            logger.error("STEP1_DATA_MISSING: No cards detected")
            raise ValueError("STEP1_DATA_MISSING: No cards detected")
        
        logger.info(f"Assembling {len(request.step1_cards)} cards")
        
        # Build index maps
        cards_by_index = {card.card_index: card for card in request.step1_cards}
        
        players_by_card_slot = {}
        for player in request.step2_players:
            key = (player.card_index, player.slot_index)
            players_by_card_slot[key] = player
        
        kills_by_card_slot = {}
        for kill in request.step3_kills:
            key = (kill.card_index, kill.slot_index)
            kills_by_card_slot[key] = kill
        
        names_by_card_slot = {}
        for name in request.step4_names:
            key = (name.card_index, name.slot_index)
            names_by_card_slot[key] = name
        
        # Assemble teams
        teams = []
        slot_positions = ["ROW1_LEFT", "ROW1_RIGHT", "ROW2_LEFT", "ROW2_RIGHT"]
        
        for card_index in sorted(cards_by_index.keys()):
            card = cards_by_index[card_index]
            logger.info(f"Processing card {card_index}")
            
            players = []
            all_confidences = []
            total_kills = 0
            
            # Process 4 slots per card
            for slot_index in range(1, 5):
                position = slot_positions[slot_index - 1]
                key = (card_index, slot_index)
                
                # Get data from maps (may be missing)
                kill_result = kills_by_card_slot.get(key)
                name_result = names_by_card_slot.get(key)
                player_boxes = players_by_card_slot.get(key)
                
                # Extract values with defaults
                name = name_result.name if name_result else ""
                name_confidence = name_result.confidence if name_result else "LOW"
                kills = kill_result.kills if kill_result else None
                kills_confidence = kill_result.confidence if kill_result else "LOW"
                
                # Validate name
                if name and len(name) > 32:
                    logger.warning(f"Card {card_index} Slot {slot_index}: Name too long, truncating")
                    name = name[:32]
                
                # Validate kills
                if kills is not None and (kills < 0 or kills > 20):
                    logger.warning(f"Card {card_index} Slot {slot_index}: Invalid kill count {kills}, setting to None")
                    kills = None
                    kills_confidence = "LOW"
                
                # Log missing data
                if not name_result:
                    logger.warning(f"Card {card_index} Slot {slot_index}: No name data")
                if not kill_result:
                    logger.warning(f"Card {card_index} Slot {slot_index}: No kill data")
                if not player_boxes:
                    logger.warning(f"Card {card_index} Slot {slot_index}: No player boxes")
                
                # Create player
                player = OcrPlayerFinal(
                    slot_index=slot_index,
                    position=position,
                    name=name,
                    name_confidence=name_confidence,
                    kills=kills,
                    kills_confidence=kills_confidence
                )
                players.append(player)
                
                # Accumulate for confidence calculation
                all_confidences.append(name_confidence)
                all_confidences.append(kills_confidence)
                
                # Add to total kills
                if kills is not None:
                    total_kills += kills
                
                logger.debug(f"Card {card_index} Slot {slot_index}: name='{name}' ({name_confidence}), kills={kills} ({kills_confidence})")
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(all_confidences)
            
            # Check team integrity
            non_empty_names = sum(1 for p in players if p.name and p.name.strip())
            non_null_kills = sum(1 for p in players if p.kills is not None)
            
            if non_empty_names == 0 and non_null_kills == 0:
                logger.warning(f"Card {card_index}: No readable names or kills, marking overall_confidence=LOW")
                overall_confidence = "LOW"
            
            # Create team
            team = OcrTeamFinal(
                card_index=card_index,
                approx_visual_rank=card.approx_visual_rank,
                players=players,
                total_kills=total_kills,
                overall_confidence=overall_confidence
            )
            teams.append(team)
            
            logger.info(f"Card {card_index}: total_kills={total_kills}, overall_confidence={overall_confidence}")
        
        # Create final result
        result = OcrMatchFinal(
            match_id=request.match_id,
            teams=teams
        )
        
        logger.info(f"Finalization complete. {len(teams)} teams assembled")
        return result
    
    def _calculate_overall_confidence(self, confidences: list[str]) -> str:
        """
        Calculate overall confidence from individual confidences.
        
        Args:
            confidences: List of confidence strings (HIGH, MEDIUM, LOW)
            
        Returns:
            Overall confidence (HIGH, MEDIUM, LOW)
        """
        if not confidences:
            return "LOW"
        
        has_low = "LOW" in confidences
        has_medium = "MEDIUM" in confidences
        all_high = all(c == "HIGH" for c in confidences)
        
        if all_high:
            return "HIGH"
        elif has_low:
            return "LOW"
        else:
            return "MEDIUM"
