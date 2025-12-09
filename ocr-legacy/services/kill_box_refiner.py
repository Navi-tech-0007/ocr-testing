import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

logger = logging.getLogger(__name__)


class KillBoxRefiner:
    """Service for refining and padding kill boxes from Step 2 player detection."""
    
    # Padding ratios (relative to card dimensions)
    # Increased to ensure minimum 30px height for OCR
    LEFT_PADDING_RATIO = 0.05
    RIGHT_PADDING_RATIO = 0.08
    TOP_PADDING_RATIO = 0.35  # Increased from 0.12
    BOTTOM_PADDING_RATIO = 0.35  # Increased from 0.12
    
    # Preview scale factor
    PREVIEW_SCALE_FACTOR = 3.0
    
    # Minimum dimensions for OCR
    MIN_WIDTH = 20
    MIN_HEIGHT = 30
    
    @staticmethod
    def refine_kill_boxes(players, card_width, card_height):
        """
        Refine kill boxes by applying padding and expansion.
        
        Args:
            players: List of player dicts from Step 2 (with raw kill_box)
            card_width: Width of the card crop
            card_height: Height of the card crop
            
        Returns:
            List of refined players with padded kill_box and metadata
        """
        refined_players = []
        
        for player in players:
            refined = player.copy()
            
            if not player.get("kill_box"):
                refined["kill_box_refined"] = None
                refined["kill_box_valid"] = False
                refined["kill_box_error"] = "No kill_box in raw data"
                refined_players.append(refined)
                continue
            
            try:
                raw_box = player["kill_box"]
                
                # Estimate row height from player position
                # Assuming 2 rows per card, each row is roughly card_height / 2
                row_height = card_height / 2
                
                # Calculate padding
                left_pad = card_width * KillBoxRefiner.LEFT_PADDING_RATIO
                right_pad = card_width * KillBoxRefiner.RIGHT_PADDING_RATIO
                top_pad = row_height * KillBoxRefiner.TOP_PADDING_RATIO
                bottom_pad = row_height * KillBoxRefiner.BOTTOM_PADDING_RATIO
                
                # Apply padding
                expanded_x1 = raw_box["x1"] - left_pad
                expanded_x2 = raw_box["x2"] + right_pad
                expanded_y1 = raw_box["y1"] - top_pad
                expanded_y2 = raw_box["y2"] + bottom_pad
                
                # Clamp to card boundaries
                expanded_x1 = max(int(expanded_x1), 0)
                expanded_y1 = max(int(expanded_y1), 0)
                expanded_x2 = min(int(expanded_x2), card_width)
                expanded_y2 = min(int(expanded_y2), card_height)
                
                # Calculate dimensions
                width = expanded_x2 - expanded_x1
                height = expanded_y2 - expanded_y1
                
                # Validate dimensions
                is_valid = width >= KillBoxRefiner.MIN_WIDTH and height >= KillBoxRefiner.MIN_HEIGHT
                
                refined["kill_box_refined"] = {
                    "x1": expanded_x1,
                    "y1": expanded_y1,
                    "x2": expanded_x2,
                    "y2": expanded_y2
                }
                
                refined["kill_box_valid"] = is_valid
                refined["kill_box_dimensions"] = {
                    "width": width,
                    "height": height,
                    "raw_width": raw_box["x2"] - raw_box["x1"],
                    "raw_height": raw_box["y2"] - raw_box["y1"]
                }
                
                if not is_valid:
                    refined["kill_box_error"] = f"Dimensions too small: {width}x{height} (min: {KillBoxRefiner.MIN_WIDTH}x{KillBoxRefiner.MIN_HEIGHT})"
                
                logger.info(f"Refined kill box: Card {player.get('card_index')} Slot {player.get('slot_index')}: {width}x{height}")
            
            except Exception as e:
                logger.error(f"Error refining kill box: {str(e)}")
                refined["kill_box_refined"] = None
                refined["kill_box_valid"] = False
                refined["kill_box_error"] = str(e)
            
            refined_players.append(refined)
        
        return refined_players
    
    @staticmethod
    def create_preview_crops(card_crop_bytes, refined_players):
        """
        Create upscaled preview crops for visualization.
        
        Args:
            card_crop_bytes: Raw card crop image bytes
            refined_players: List of refined players with padded kill_box
            
        Returns:
            List of preview crop dicts with base64 images
        """
        try:
            # Decode card crop
            card_img = Image.open(BytesIO(card_crop_bytes))
            card_array = np.array(card_img)
            
            preview_crops = []
            
            for player in refined_players:
                if not player.get("kill_box_refined"):
                    preview_crops.append({
                        "card_index": player.get("card_index"),
                        "slot_index": player.get("slot_index"),
                        "error": player.get("kill_box_error", "No refined box")
                    })
                    continue
                
                try:
                    box = player["kill_box_refined"]
                    
                    # Crop the region
                    crop = card_array[box["y1"]:box["y2"], box["x1"]:box["x2"]]
                    
                    if crop.size == 0:
                        preview_crops.append({
                            "card_index": player.get("card_index"),
                            "slot_index": player.get("slot_index"),
                            "error": "Empty crop region"
                        })
                        continue
                    
                    # Upscale for preview using bicubic interpolation
                    crop_pil = Image.fromarray(crop)
                    new_width = int(crop_pil.width * KillBoxRefiner.PREVIEW_SCALE_FACTOR)
                    new_height = int(crop_pil.height * KillBoxRefiner.PREVIEW_SCALE_FACTOR)
                    crop_upscaled = crop_pil.resize((new_width, new_height), Image.BICUBIC)
                    
                    # Convert to base64
                    crop_output = BytesIO()
                    crop_upscaled.save(crop_output, format='JPEG', quality=95)
                    crop_b64 = base64.b64encode(crop_output.getvalue()).decode('utf-8')
                    
                    preview_crop = {
                        "card_index": player.get("card_index"),
                        "slot_index": player.get("slot_index"),
                        "crop_base64": crop_b64,
                        "width": player["kill_box_dimensions"]["width"],
                        "height": player["kill_box_dimensions"]["height"],
                        "raw_width": player["kill_box_dimensions"]["raw_width"],
                        "raw_height": player["kill_box_dimensions"]["raw_height"],
                        "valid": player.get("kill_box_valid", False)
                    }
                    
                    if player.get("kill_box_error"):
                        preview_crop["error"] = player["kill_box_error"]
                    
                    preview_crops.append(preview_crop)
                    logger.info(f"Created preview crop: Card {player.get('card_index')} Slot {player.get('slot_index')}")
                
                except Exception as e:
                    logger.error(f"Error creating preview crop: {str(e)}")
                    preview_crops.append({
                        "card_index": player.get("card_index"),
                        "slot_index": player.get("slot_index"),
                        "error": str(e)
                    })
            
            return preview_crops
        
        except Exception as e:
            logger.exception(f"Error creating preview crops: {str(e)}")
            raise ValueError(f"Failed to create preview crops: {str(e)}")
