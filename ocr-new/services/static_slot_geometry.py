import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

logger = logging.getLogger(__name__)


class StaticSlotGeometry:
    """
    Compute player slots and kill boxes using static geometry ratios.
    No Claude Vision, no ML — pure coordinate math based on card bounds.
    """

    # X-ranges for left and right columns (relative to card width)
    LEFT_COL_X1 = 0.05
    LEFT_COL_X2 = 0.45
    RIGHT_COL_X1 = 0.55
    RIGHT_COL_X2 = 0.95

    # Y-ranges for row 1 and row 2 (relative to card height)
    ROW1_Y1 = 0.05
    ROW1_Y2 = 0.50
    ROW2_Y1 = 0.50
    ROW2_Y2 = 0.95

    # Inside each row-column box, split into name zone and kill zone
    NAME_ZONE_X1 = 0.00
    NAME_ZONE_X2 = 0.65
    KILL_ZONE_X1 = 0.65
    KILL_ZONE_X2 = 1.00

    # Extra padding for kill boxes (relative to card/row dimensions)
    KILL_LEFT_PAD_RATIO = 0.05    # 5% of card width (more generous)
    KILL_RIGHT_PAD_RATIO = 0.05   # 5% of card width (more generous)
    KILL_TOP_PAD_RATIO = 0.20     # 20% of row height (more generous)
    KILL_BOTTOM_PAD_RATIO = 0.20  # 20% of row height (more generous)

    # Minimum kill box dimensions
    MIN_KILL_WIDTH = 15
    MIN_KILL_HEIGHT = 15

    # Preview scale factor
    PREVIEW_SCALE_FACTOR = 3.0

    def __init__(self):
        """Initialize the static geometry service."""
        logger.info("StaticSlotGeometry initialized")

    def compute_slots_and_kill_boxes(
        self,
        cards: list,
        screenshot_width: int,
        screenshot_height: int
    ) -> list:
        """
        Compute player slots and kill boxes using static geometry.

        Args:
            cards: List of card dicts with card_index and bounds
            screenshot_width: Full screenshot width in pixels
            screenshot_height: Full screenshot height in pixels

        Returns:
            List of player dicts with card_index, slot_index, position, name_box, kill_box
        """
        players = []

        for card in cards:
            card_index = card.get("card_index")
            bounds = card.get("bounds")

            if not bounds:
                logger.warning(f"Card {card_index}: No bounds found")
                continue

            try:
                card_x1 = bounds["x1"]
                card_y1 = bounds["y1"]
                card_x2 = bounds["x2"]
                card_y2 = bounds["y2"]

                card_w = card_x2 - card_x1
                card_h = card_y2 - card_y1

                if card_w <= 0 or card_h <= 0:
                    logger.warning(f"Card {card_index}: Invalid dimensions {card_w}x{card_h}")
                    continue

                # Define row boxes (absolute coordinates)
                row1_y1 = card_y1 + int(card_h * self.ROW1_Y1)
                row1_y2 = card_y1 + int(card_h * self.ROW1_Y2)
                row2_y1 = card_y1 + int(card_h * self.ROW2_Y1)
                row2_y2 = card_y1 + int(card_h * self.ROW2_Y2)

                # Define column boxes (absolute coordinates)
                left_x1 = card_x1 + int(card_w * self.LEFT_COL_X1)
                left_x2 = card_x1 + int(card_w * self.LEFT_COL_X2)
                right_x1 = card_x1 + int(card_w * self.RIGHT_COL_X1)
                right_x2 = card_x1 + int(card_w * self.RIGHT_COL_X2)

                # Slot definitions: (row_y1, row_y2, col_x1, col_x2, position, slot_index)
                slots_config = [
                    (row1_y1, row1_y2, left_x1, left_x2, "ROW1_LEFT", 1),
                    (row1_y1, row1_y2, right_x1, right_x2, "ROW1_RIGHT", 2),
                    (row2_y1, row2_y2, left_x1, left_x2, "ROW2_LEFT", 3),
                    (row2_y1, row2_y2, right_x1, right_x2, "ROW2_RIGHT", 4),
                ]

                for row_y1, row_y2, col_x1, col_x2, position, slot_index in slots_config:
                    try:
                        col_w = col_x2 - col_x1
                        row_h = row_y2 - row_y1

                        # Name box (left portion of the column)
                        name_x1 = col_x1 + int(col_w * self.NAME_ZONE_X1)
                        name_x2 = col_x1 + int(col_w * self.NAME_ZONE_X2)
                        name_y1 = row_y1
                        name_y2 = row_y2

                        # Raw kill box (right portion of the column)
                        kill_raw_x1 = col_x1 + int(col_w * self.KILL_ZONE_X1)
                        kill_raw_x2 = col_x1 + int(col_w * self.KILL_ZONE_X2)
                        kill_raw_y1 = row_y1
                        kill_raw_y2 = row_y2

                        # Apply padding to kill box
                        kill_x1 = max(
                            int(kill_raw_x1 - card_w * self.KILL_LEFT_PAD_RATIO),
                            card_x1
                        )
                        kill_x2 = min(
                            int(kill_raw_x2 + card_w * self.KILL_RIGHT_PAD_RATIO),
                            card_x2
                        )
                        kill_y1 = max(
                            int(kill_raw_y1 - row_h * self.KILL_TOP_PAD_RATIO),
                            card_y1
                        )
                        kill_y2 = min(
                            int(kill_raw_y2 + row_h * self.KILL_BOTTOM_PAD_RATIO),
                            card_y2
                        )

                        # Clamp to screenshot bounds
                        kill_x1 = max(kill_x1, 0)
                        kill_y1 = max(kill_y1, 0)
                        kill_x2 = min(kill_x2, screenshot_width)
                        kill_y2 = min(kill_y2, screenshot_height)

                        kill_width = kill_x2 - kill_x1
                        kill_height = kill_y2 - kill_y1

                        # Validate minimum dimensions
                        is_valid = (
                            kill_width >= self.MIN_KILL_WIDTH
                            and kill_height >= self.MIN_KILL_HEIGHT
                        )

                        player = {
                            "card_index": card_index,
                            "slot_index": slot_index,
                            "position": position,
                            "name_box": {
                                "x1": max(name_x1, 0),
                                "y1": max(name_y1, 0),
                                "x2": min(name_x2, screenshot_width),
                                "y2": min(name_y2, screenshot_height),
                            },
                            "kill_box": {
                                "x1": kill_x1,
                                "y1": kill_y1,
                                "x2": kill_x2,
                                "y2": kill_y2,
                            },
                            "kill_box_dimensions": {
                                "width": kill_width,
                                "height": kill_height,
                            },
                            "kill_box_valid": is_valid,
                        }

                        if not is_valid:
                            player["kill_box_error"] = (
                                f"Dimensions too small: {kill_width}x{kill_height} "
                                f"(min: {self.MIN_KILL_WIDTH}x{self.MIN_KILL_HEIGHT})"
                            )

                        players.append(player)
                        logger.debug(
                            f"Card {card_index} {position} (slot {slot_index}): "
                            f"kill_box {kill_width}x{kill_height}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Card {card_index} {position} (slot {slot_index}): {str(e)}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Card {card_index}: {str(e)}")
                continue

        logger.info(f"Computed {len(players)} player slots from {len(cards)} cards")
        return players

    def create_kill_preview_crops(
        self,
        screenshot_array: np.ndarray,
        players: list
    ) -> list:
        """
        Create upscaled preview crops of kill boxes for visualization.

        Args:
            screenshot_array: Full screenshot as numpy array
            players: List of player dicts with kill_box coordinates

        Returns:
            List of preview crop dicts with base64 images
        """
        preview_crops = []

        for player in players:
            try:
                card_index = player.get("card_index")
                slot_index = player.get("slot_index")
                kill_box = player.get("kill_box")

                if not kill_box:
                    preview_crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "error": "No kill_box found",
                    })
                    continue

                # Extract kill box coordinates
                x1 = int(kill_box["x1"])
                y1 = int(kill_box["y1"])
                x2 = int(kill_box["x2"])
                y2 = int(kill_box["y2"])

                # Validate coordinates
                if x1 >= x2 or y1 >= y2:
                    preview_crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "error": "Invalid box coordinates",
                    })
                    continue

                # Crop from screenshot
                crop = screenshot_array[y1:y2, x1:x2]

                if crop.size == 0:
                    preview_crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "error": "Empty crop",
                    })
                    continue

                # Upscale crop
                crop_h, crop_w = crop.shape[:2]
                new_w = int(crop_w * self.PREVIEW_SCALE_FACTOR)
                new_h = int(crop_h * self.PREVIEW_SCALE_FACTOR)

                if new_w > 0 and new_h > 0:
                    crop_upscaled = cv2.resize(
                        crop,
                        (new_w, new_h),
                        interpolation=cv2.INTER_CUBIC
                    )
                else:
                    crop_upscaled = crop

                # Convert to PIL Image and encode to base64
                if len(crop_upscaled.shape) == 2:
                    # Grayscale
                    crop_img = Image.fromarray(crop_upscaled, mode='L')
                else:
                    # Color
                    crop_img = Image.fromarray(crop_upscaled)

                crop_output = BytesIO()
                crop_img.save(crop_output, format='JPEG', quality=95)
                crop_b64 = base64.b64encode(crop_output.getvalue()).decode('utf-8')

                preview_crops.append({
                    "card_index": card_index,
                    "slot_index": slot_index,
                    "crop_base64": crop_b64,
                    "width": new_w,
                    "height": new_h,
                })

                logger.debug(
                    f"Card {card_index} Slot {slot_index}: "
                    f"preview crop {new_w}x{new_h}"
                )

            except Exception as e:
                logger.error(
                    f"Error creating preview for card {player.get('card_index')} "
                    f"slot {player.get('slot_index')}: {str(e)}"
                )
                preview_crops.append({
                    "card_index": player.get("card_index"),
                    "slot_index": player.get("slot_index"),
                    "error": str(e),
                })
                continue

        logger.info(f"Created {len(preview_crops)} preview crops")
        return preview_crops
