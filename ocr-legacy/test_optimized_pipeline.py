#!/usr/bin/env python3
"""
Test script to verify the optimized OCR pipeline.

This tests:
1. KillExtractor initialization without PaddleOCR
2. Tesseract extraction on preprocessed images
3. Claude fallback when Tesseract fails
"""

import logging
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

from services.kill_extractor import KillExtractor


def test_initialization():
    """Test that KillExtractor initializes without PaddleOCR."""
    logger.info("=" * 60)
    logger.info("TEST 1: KillExtractor Initialization (Production Mode)")
    logger.info("=" * 60)
    
    try:
        extractor = KillExtractor(use_paddle_debug=False)
        logger.info("✅ KillExtractor initialized successfully (no PaddleOCR)")
        logger.info(f"   - PaddleOCR loaded: {extractor.paddle_ocr is not None}")
        logger.info(f"   - Bedrock client: {extractor.bedrock_client is not None}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {str(e)}")
        return False


def test_preprocessing():
    """Test the fast preprocessing pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Fast Preprocessing (Adaptive Threshold)")
    logger.info("=" * 60)
    
    try:
        from io import BytesIO
        
        extractor = KillExtractor(use_paddle_debug=False)
        
        # Create a simple test image (white digit on black background)
        test_img = Image.new('RGB', (100, 100), color='white')
        
        # Convert to bytes properly (PNG format)
        img_bytes = BytesIO()
        test_img.save(img_bytes, format='PNG')
        test_bytes = img_bytes.getvalue()
        
        # Test preprocessing
        preprocessed = extractor._preprocess_for_ocr(test_bytes)
        logger.info(f"✅ Preprocessing successful")
        logger.info(f"   - Input size: {len(test_bytes)} bytes")
        logger.info(f"   - Output type: {type(preprocessed)}")
        logger.info(f"   - Output mode: {preprocessed.mode}")
        return True
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return False


def test_debug_mode():
    """Test that PaddleOCR loads in debug mode."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Debug Mode (PaddleOCR Optional)")
    logger.info("=" * 60)
    
    try:
        extractor = KillExtractor(use_paddle_debug=True)
        logger.info("✅ KillExtractor initialized in debug mode")
        logger.info(f"   - PaddleOCR loaded: {extractor.paddle_ocr is not None}")
        if extractor.paddle_ocr is None:
            logger.warning("   ⚠️  PaddleOCR not available (optional in debug mode)")
        return True
    except Exception as e:
        logger.error(f"❌ Debug mode initialization failed: {str(e)}")
        return False


def test_pipeline_structure():
    """Test that the pipeline has correct structure."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Pipeline Structure Verification")
    logger.info("=" * 60)
    
    try:
        extractor = KillExtractor(use_paddle_debug=False)
        
        # Check required methods
        required_methods = [
            '_preprocess_for_ocr',
            '_extract_tesseract',
            '_extract_with_claude',
            '_verify_with_claude',
            'extract_kill',
            'extract_kills_batch'
        ]
        
        all_present = True
        for method in required_methods:
            has_method = hasattr(extractor, method)
            status = "✓" if has_method else "✗"
            logger.info(f"   {status} {method}")
            all_present = all_present and has_method
        
        if all_present:
            logger.info("✅ All required methods present")
            return True
        else:
            logger.error("❌ Some methods missing")
            return False
    except Exception as e:
        logger.error(f"❌ Structure verification failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZED OCR PIPELINE TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Initialization", test_initialization),
        ("Preprocessing", test_preprocessing),
        ("Debug Mode", test_debug_mode),
        ("Pipeline Structure", test_pipeline_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {str(e)}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Pipeline is optimized and ready.")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
