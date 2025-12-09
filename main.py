import base64
import uuid
import logging
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from services.card_detector import TeamCardDetector, CardDetectionError
from services.player_detector import PlayerSlotDetector, PlayerDetectionError
from services.kill_extractor import KillExtractor, KillExtractionError
from services.name_extractor import NameExtractor, NameExtractionError
from services.finalizer import OcrFinalizer, OcrFinalizationRequest
from services.image_cropper import ImageCropper
from services.kill_box_refiner import KillBoxRefiner
from services.ocr_debugger import OcrDebugger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Team Card Detection Service",
    description="Detects team card bounding boxes from Free Fire scoreboard screenshots",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
card_detector = TeamCardDetector()
player_detector = PlayerSlotDetector()
kill_extractor = KillExtractor()
name_extractor = NameExtractor()
finalizer = OcrFinalizer()
ocr_debugger = OcrDebugger()

# Global storage for original full-resolution screenshot and card boxes (used by debug mode)
global_original_screenshot_array = None
global_original_screenshot_base64 = None
global_card_boxes = {}  # {card_index: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}}


@app.post("/ocr/cards/detect")
async def detect_team_cards(
    file: UploadFile = File(...),
    request_id: str = Form(default=None),
    game_metadata: str = Form(default=None),
):
    """
    Detect team card bounding boxes from a Free Fire scoreboard screenshot.
    
    Args:
        file: Multipart image upload (image/*)
        request_id: Optional request identifier
        game_metadata: Optional game metadata (ignored for main logic)
    
    Returns:
        JSON response with detected cards and their bounding boxes
    """
    # Generate request_id if not provided
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] POST /ocr/cards/detect - File: {file.filename}")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"[{request_id}] Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="File must be an image (image/*)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if not image_bytes:
            logger.warning(f"[{request_id}] Empty image file")
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )
        
        logger.info(f"[{request_id}] Image read. Size: {len(image_bytes)} bytes")
        
        # CRITICAL: Store original screenshot for debug mode
        global global_original_screenshot_array, global_original_screenshot_base64
        from PIL import Image
        import numpy as np
        
        screenshot_img = Image.open(BytesIO(image_bytes))
        global_original_screenshot_array = np.array(screenshot_img)
        global_original_screenshot_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.info(f"[{request_id}] Stored original screenshot: {global_original_screenshot_array.shape}")
        print(f"[DEBUG] STORED ORIGINAL IMAGE SHAPE: {global_original_screenshot_array.shape}")
        
        # Detect cards
        result = card_detector.detect_cards(image_bytes)
        
        # CRITICAL: Store card bounding boxes for debug mode coordinate conversion
        global global_card_boxes
        global_card_boxes = {}
        for card in result.get('cards', []):
            card_index = card.get('card_index')
            bounds = card.get('bounds')
            if card_index and bounds:
                global_card_boxes[card_index] = bounds
                logger.info(f"[{request_id}] Stored card {card_index} bounds: {bounds}")
        
        print(f"[DEBUG] STORED CARD BOXES: {global_card_boxes}")
        
        # Add request_id to response
        result["request_id"] = request_id
        
        logger.info(f"[{request_id}] Detection successful. Cards: {len(result['cards'])}")
        return JSONResponse(status_code=200, content=result)
    
    except CardDetectionError as e:
        logger.error(f"[{request_id}] Detection error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "cards": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "cards": []
            }
        )


@app.post("/ocr/cards/players/detect")
async def detect_all_players(request_data: dict):
    """
    Detect player slots from all team cards (Step 1 output).
    
    Args:
        request_data: Dict with screenshot_base64 and step1_cards
        
    Returns:
        JSON response with all player slot detections
    """
    request_id = str(uuid.uuid4())
    
    try:
        screenshot_base64 = request_data.get("screenshot_base64")
        step1_cards = request_data.get("step1_cards", [])
        
        if not screenshot_base64:
            logger.error(f"[{request_id}] Missing screenshot_base64")
            raise ValueError("screenshot_base64 required")
        
        if not step1_cards:
            logger.error(f"[{request_id}] No cards from Step 1")
            raise ValueError("step1_cards required")
        
        logger.info(f"[{request_id}] Processing {len(step1_cards)} cards for player detection")
        
        # Decode screenshot
        import base64
        screenshot_bytes = base64.b64decode(screenshot_base64)
        
        all_players = []
        
        # Process each card
        for card in step1_cards:
            card_index = card.get("card_index")
            bounds = card.get("bounds")
            
            try:
                # Crop card from screenshot
                card_crop = ImageCropper.crop_image(
                    screenshot_bytes,
                    bounds["x1"], bounds["y1"],
                    bounds["x2"], bounds["y2"]
                )
                
                logger.info(f"[{request_id}] Card {card_index}: cropped {len(card_crop)} bytes")
                
                # Detect players in this card
                result = player_detector.detect_players(card_crop)
                
                # Add card_index to each player
                if "players" in result:
                    for player in result["players"]:
                        player["card_index"] = card_index
                    all_players.extend(result["players"])
                
                logger.info(f"[{request_id}] Card {card_index}: detected {len(result.get('players', []))} players")
            
            except Exception as e:
                logger.error(f"[{request_id}] Card {card_index} error: {str(e)}")
                # Continue with next card
                continue
        
        logger.info(f"[{request_id}] Player detection complete. Total players: {len(all_players)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "players": all_players
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "players": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "players": []
            }
        )


@app.post("/ocr/cards/{card_index}/players/detect")
async def detect_players(
    card_index: int,
    file: UploadFile = File(...),
    request_id: str = Form(default=None),
):
    """
    Detect player slots from a cropped team card image.
    
    Args:
        card_index: Card index from Step 1
        file: Multipart image upload (cropped team card)
        request_id: Optional request identifier
    
    Returns:
        JSON response with player slot bounding boxes
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] POST /ocr/cards/{card_index}/players/detect - File: {file.filename}")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"[{request_id}] Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="File must be an image (image/*)"
        )
    
    try:
        image_bytes = await file.read()
        
        if not image_bytes:
            logger.warning(f"[{request_id}] Empty image file")
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )
        
        logger.info(f"[{request_id}] Image read. Size: {len(image_bytes)} bytes")
        
        # Detect players
        result = player_detector.detect_players(image_bytes)
        
        # Add metadata to response
        result["request_id"] = request_id
        result["card_index"] = card_index
        
        logger.info(f"[{request_id}] Detection successful. Players: {len(result['players'])}")
        return JSONResponse(status_code=200, content=result)
    
    except PlayerDetectionError as e:
        logger.error(f"[{request_id}] Detection error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "card_index": card_index,
                "error": str(e),
                "players": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "card_index": card_index,
                "error": f"Internal server error: {str(e)}",
                "players": []
            }
        )


@app.post("/ocr/players/refine-kill-boxes")
async def refine_kill_boxes(request_data: dict):
    """
    Step 2.5: Refine and pad kill boxes from Step 2 player detection.
    
    Args:
        request_data: Dict with players (from Step 2), card_crop_base64, card dimensions
        
    Returns:
        JSON response with refined players and preview crops
    """
    request_id = str(uuid.uuid4())
    
    try:
        import base64
        
        players = request_data.get("players", [])
        card_crop_b64 = request_data.get("card_crop")
        card_width = request_data.get("card_width")
        card_height = request_data.get("card_height")
        
        if not players:
            logger.warning(f"[{request_id}] No players provided")
            raise ValueError("players list required")
        
        if not card_crop_b64:
            logger.warning(f"[{request_id}] No card crop provided")
            raise ValueError("card_crop required")
        
        if not card_width or not card_height:
            logger.warning(f"[{request_id}] Card dimensions required")
            raise ValueError("card_width and card_height required")
        
        logger.info(f"[{request_id}] Refining {len(players)} kill boxes (card: {card_width}x{card_height})")
        
        # Refine kill boxes with padding
        refined_players = KillBoxRefiner.refine_kill_boxes(players, card_width, card_height)
        
        # Decode card crop for preview generation
        card_crop_bytes = base64.b64decode(card_crop_b64)
        
        # Create upscaled preview crops
        preview_crops = KillBoxRefiner.create_preview_crops(card_crop_bytes, refined_players)
        
        # Check if all crops are valid
        valid_count = sum(1 for p in preview_crops if p.get("valid", False))
        
        logger.info(f"[{request_id}] Refinement complete. Valid crops: {valid_count}/{len(preview_crops)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "refined_players": refined_players,
                "preview_crops": preview_crops,
                "valid_count": valid_count,
                "total_count": len(preview_crops)
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "refined_players": [],
                "preview_crops": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "refined_players": [],
                "preview_crops": []
            }
        )


@app.post("/ocr/kills/debug-crops")
async def debug_kill_crops(request_data: dict):
    """
    Debug endpoint: crop kill boxes from card crops WITHOUT running OCR.
    
    Args:
        request_data: Dict with kill_boxes list (each has card_index, slot_index, box, card_crop_base64)
        
    Returns:
        JSON response with cropped images and metadata
    """
    request_id = str(uuid.uuid4())
    
    try:
        import base64
        import numpy as np
        from PIL import Image
        
        kill_boxes = request_data.get("kill_boxes", [])
        
        if not kill_boxes:
            logger.warning(f"[{request_id}] No kill boxes provided")
            raise ValueError("kill_boxes list required")
        
        logger.info(f"[{request_id}] Debugging {len(kill_boxes)} kill boxes")
        
        crops = []
        
        for kb in kill_boxes:
            try:
                card_index = kb.get("card_index")
                slot_index = kb.get("slot_index")
                box = kb.get("box")
                card_crop_b64 = kb.get("card_crop")
                
                if not all([card_index, slot_index, box, card_crop_b64]):
                    logger.error(f"[{request_id}] Missing fields in kill box")
                    continue
                
                # Decode card crop
                card_crop_bytes = base64.b64decode(card_crop_b64)
                card_img = Image.open(BytesIO(card_crop_bytes))
                card_width, card_height = card_img.size
                
                # Extract box coordinates (relative to card crop)
                x1 = int(box.get("x1", 0))
                y1 = int(box.get("y1", 0))
                x2 = int(box.get("x2", 0))
                y2 = int(box.get("y2", 0))
                
                # Validate coordinates
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"[{request_id}] Card {card_index} Slot {slot_index}: Invalid box coordinates")
                    crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "error": "INVALID_BOX"
                    })
                    continue
                
                if x1 < 0 or y1 < 0 or x2 > card_width or y2 > card_height:
                    logger.warning(f"[{request_id}] Card {card_index} Slot {slot_index}: Box out of bounds")
                    crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "error": "INVALID_BOX"
                    })
                    continue
                
                # Crop the kill box
                crop_img = card_img.crop((x1, y1, x2, y2))
                crop_width, crop_height = crop_img.size
                
                # Check if crop is too small
                if crop_width < 5 or crop_height < 5:
                    logger.warning(f"[{request_id}] Card {card_index} Slot {slot_index}: Crop too small ({crop_width}x{crop_height})")
                    crops.append({
                        "card_index": card_index,
                        "slot_index": slot_index,
                        "width": crop_width,
                        "height": crop_height,
                        "error": "INVALID_BOX"
                    })
                    continue
                
                # Check if crop is mostly empty (dark/blank)
                crop_array = np.array(crop_img.convert('L'))
                mean_pixel = float(np.mean(crop_array))
                
                warning = None
                if mean_pixel < 10:
                    warning = "EMPTY_CROP"
                    logger.warning(f"[{request_id}] Card {card_index} Slot {slot_index}: Empty/dark crop (mean={mean_pixel:.1f})")
                
                # Convert crop to base64
                crop_output = BytesIO()
                crop_img.save(crop_output, format='JPEG', quality=95)
                crop_b64 = base64.b64encode(crop_output.getvalue()).decode('utf-8')
                
                crop_result = {
                    "card_index": card_index,
                    "slot_index": slot_index,
                    "width": crop_width,
                    "height": crop_height,
                    "crop_base64": crop_b64
                }
                
                if warning:
                    crop_result["warning"] = warning
                
                crops.append(crop_result)
                logger.info(f"[{request_id}] Card {card_index} Slot {slot_index}: Cropped {crop_width}x{crop_height}")
            
            except Exception as e:
                logger.error(f"[{request_id}] Error processing kill box: {str(e)}")
                crops.append({
                    "card_index": kb.get("card_index"),
                    "slot_index": kb.get("slot_index"),
                    "error": str(e)
                })
                continue
        
        logger.info(f"[{request_id}] Debug complete. Crops: {len(crops)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "crops": crops
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "crops": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "crops": []
            }
        )


@app.post("/ocr/kills/extract")
async def extract_kills(request_data: dict):
    """
    Extract kill counts from kill box coordinates (Step 3).
    
    Args:
        request_data: Dict with kill_crops (kill_box coordinates) and screenshot_base64
        
    Returns:
        JSON response with extraction results
    """
    request_id = str(uuid.uuid4())
    
    try:
        import base64
        
        kill_crops = request_data.get("kill_crops", [])
        screenshot_base64 = request_data.get("screenshot_base64")
        
        if not kill_crops:
            logger.warning(f"[{request_id}] No kill crops provided")
            raise ValueError("kill_crops list required")
        
        if not screenshot_base64:
            logger.warning(f"[{request_id}] No screenshot provided")
            raise ValueError("screenshot_base64 required")
        
        logger.info(f"[{request_id}] Processing {len(kill_crops)} kill boxes")
        
        # Decode screenshot
        screenshot_bytes = base64.b64decode(screenshot_base64)
        
        # Crop kill boxes and extract
        kill_crops_with_images = []
        for crop in kill_crops:
            try:
                # Crop kill_box from screenshot
                kill_crop_bytes = ImageCropper.crop_image(
                    screenshot_bytes,
                    crop["kill_box"]["x1"],
                    crop["kill_box"]["y1"],
                    crop["kill_box"]["x2"],
                    crop["kill_box"]["y2"]
                )
                
                # Convert to base64
                kill_crop_b64 = base64.b64encode(kill_crop_bytes).decode('utf-8')
                
                kill_crops_with_images.append({
                    "card_index": crop["card_index"],
                    "slot_index": crop["slot_index"],
                    "image": kill_crop_b64
                })
                
                logger.debug(f"[{request_id}] Card {crop['card_index']} Slot {crop['slot_index']}: cropped kill box")
            
            except Exception as e:
                logger.error(f"[{request_id}] Failed to crop kill box: {str(e)}")
                continue
        
        if not kill_crops_with_images:
            logger.error(f"[{request_id}] No kill crops could be extracted")
            raise ValueError("Failed to crop any kill boxes")
        
        logger.info(f"[{request_id}] Extracted {len(kill_crops_with_images)} kill crop images")
        
        # Extract kills using PaddleOCR + Tesseract + Claude
        results = kill_extractor.extract_kills_batch(kill_crops_with_images)
        
        logger.info(f"[{request_id}] Kill extraction complete. Results: {len(results)}")
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "results": results
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "results": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "results": []
            }
        )


@app.post("/ocr/names/extract")
async def extract_names(request_data: dict):
    """
    Extract player names from name box coordinates (Step 4).
    
    Args:
        request_data: Dict with name_crops (name_box coordinates) and screenshot_base64
        
    Returns:
        JSON response with extraction results
    """
    request_id = str(uuid.uuid4())
    
    try:
        import base64
        
        name_crops = request_data.get("name_crops", [])
        screenshot_base64 = request_data.get("screenshot_base64")
        
        if not name_crops:
            logger.warning(f"[{request_id}] No name crops provided")
            raise ValueError("name_crops list required")
        
        if not screenshot_base64:
            logger.warning(f"[{request_id}] No screenshot provided")
            raise ValueError("screenshot_base64 required")
        
        logger.info(f"[{request_id}] Processing {len(name_crops)} name boxes")
        
        # Decode screenshot
        screenshot_bytes = base64.b64decode(screenshot_base64)
        
        # Crop name boxes and extract
        name_crops_with_images = []
        for crop in name_crops:
            try:
                # Crop name_box from screenshot
                name_crop_bytes = ImageCropper.crop_image(
                    screenshot_bytes,
                    crop["name_box"]["x1"],
                    crop["name_box"]["y1"],
                    crop["name_box"]["x2"],
                    crop["name_box"]["y2"]
                )
                
                # Convert to base64
                name_crop_b64 = base64.b64encode(name_crop_bytes).decode('utf-8')
                
                name_crops_with_images.append({
                    "card_index": crop["card_index"],
                    "slot_index": crop["slot_index"],
                    "image": name_crop_b64
                })
                
                logger.debug(f"[{request_id}] Card {crop['card_index']} Slot {crop['slot_index']}: cropped name box")
            
            except Exception as e:
                logger.error(f"[{request_id}] Failed to crop name box: {str(e)}")
                continue
        
        if not name_crops_with_images:
            logger.error(f"[{request_id}] No name crops could be extracted")
            raise ValueError("Failed to crop any name boxes")
        
        logger.info(f"[{request_id}] Extracted {len(name_crops_with_images)} name crop images")
        
        # Extract names using Claude Vision
        results = name_extractor.extract_names_batch(name_crops_with_images)
        
        logger.info(f"[{request_id}] Name extraction complete. Results: {len(results)}")
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "results": results
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e),
                "results": []
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}",
                "results": []
            }
        )


@app.post("/ocr/finalize")
async def finalize_ocr(request_data: dict):
    """
    Assemble and validate OCR results from all 4 steps.
    
    Args:
        request_data: Dict with step1_cards, step2_players, step3_kills, step4_names
        
    Returns:
        JSON response with finalized match structure
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] Finalizing OCR results")
        
        # Parse request
        finalization_request = OcrFinalizationRequest(**request_data)
        
        logger.info(f"[{request_id}] Input validation passed")
        logger.info(f"[{request_id}] Cards: {len(finalization_request.step1_cards)}, "
                   f"Players: {len(finalization_request.step2_players)}, "
                   f"Kills: {len(finalization_request.step3_kills)}, "
                   f"Names: {len(finalization_request.step4_names)}")
        
        # Finalize
        result = finalizer.finalize(finalization_request)
        
        logger.info(f"[{request_id}] Finalization successful. Teams: {len(result.teams)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "match_id": result.match_id,
                "teams": [team.dict() for team in result.teams]
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        error_code = str(e).split(":")[0] if ":" in str(e) else "OCR_FINALIZE_INVALID_INPUT"
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error_code": error_code,
                "message": str(e)
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error_code": "OCR_FINALIZE_ERROR",
                "message": f"Internal server error: {str(e)}"
            }
        )


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    from fastapi.responses import FileResponse
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"error": "favicon not found"}


@app.post("/ocr/debug/single-slot")
async def debug_single_slot(request_data: dict):
    """
    Debug OCR for a single slot with full intermediate image visibility.
    
    Args:
        request_data: Dict with card_index, slot_index, kill_box
        (screenshot is loaded from global storage, NOT from request)
        
    Returns:
        JSON response with debug info and intermediate images
    """
    request_id = str(uuid.uuid4())
    
    try:
        card_index = request_data.get("card_index")
        slot_index = request_data.get("slot_index")
        kill_box = request_data.get("kill_box")
        
        if not all([card_index, slot_index, kill_box]):
            logger.warning(f"[{request_id}] Missing required parameters")
            raise ValueError("card_index, slot_index, and kill_box required")
        
        # CRITICAL: Use stored original screenshot, NOT request parameter
        global global_original_screenshot_array, global_original_screenshot_base64
        
        if global_original_screenshot_array is None:
            logger.error(f"[{request_id}] Original screenshot not stored. Did you run Step 1?")
            raise ValueError("Debug OCR attempted before image upload. Run Step 1 first.")
        
        logger.info(f"[{request_id}] Debug request: Card {card_index} Slot {slot_index}")
        logger.info(f"[{request_id}] Using stored original screenshot: {global_original_screenshot_array.shape}")
        print(f"[DEBUG] USING STORED ORIGINAL IMAGE SHAPE: {global_original_screenshot_array.shape}")
        
        # Call debugger with stored original screenshot and card boxes
        result = ocr_debugger.debug_single_slot(card_index, slot_index, kill_box, global_original_screenshot_array, global_card_boxes)
        
        logger.info(f"[{request_id}] Debug complete for Card {card_index} Slot {slot_index}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "result": result
            }
        )
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": f"Internal server error: {str(e)}"
            }
        )


@app.post("/ocr/debug/restore-context")
async def restore_debug_context(request_data: dict):
    """
    Restore screenshot and card boxes for debug mode without re-running Step 1.
    Used when loading cached results from sessionStorage.
    
    Args:
        request_data: Dict with screenshot_base64 and cards (from cached Step 1)
    """
    request_id = str(uuid.uuid4())
    
    try:
        screenshot_base64 = request_data.get("screenshot_base64")
        cards = request_data.get("cards", [])
        
        if not screenshot_base64:
            raise ValueError("screenshot_base64 required")
        
        # Restore global screenshot
        global global_original_screenshot_array, global_original_screenshot_base64, global_card_boxes
        from PIL import Image
        import numpy as np
        
        screenshot_bytes = base64.b64decode(screenshot_base64)
        screenshot_img = Image.open(BytesIO(screenshot_bytes))
        global_original_screenshot_array = np.array(screenshot_img)
        global_original_screenshot_base64 = screenshot_base64
        
        logger.info(f"[{request_id}] Restored screenshot: {global_original_screenshot_array.shape}")
        print(f"[DEBUG] RESTORED ORIGINAL IMAGE SHAPE: {global_original_screenshot_array.shape}")
        
        # Restore card boxes
        global_card_boxes = {}
        for card in cards:
            card_index = card.get('card_index')
            bounds = card.get('bounds')
            if card_index and bounds:
                global_card_boxes[card_index] = bounds
        
        logger.info(f"[{request_id}] Restored {len(global_card_boxes)} card boxes")
        print(f"[DEBUG] RESTORED CARD BOXES: {global_card_boxes}")
        
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "status": "ok",
                "screenshot_shape": list(global_original_screenshot_array.shape),
                "card_count": len(global_card_boxes)
            }
        )
    
    except Exception as e:
        logger.exception(f"[{request_id}] Error restoring context: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": str(e)
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Team Card Detection Service",
        "version": "1.0.0",
        "endpoint": "POST /ocr/cards/detect",
        "health": "GET /health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
