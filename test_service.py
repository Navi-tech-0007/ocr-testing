#!/usr/bin/env python3
"""
Test script for Team Card Detection Service.
Tests the service with a sample image.
"""

import requests
import sys
from pathlib import Path


def test_health():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_root():
    """Test root endpoint."""
    print("Testing root endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_card_detection(image_path: str):
    """
    Test card detection endpoint.
    
    Args:
        image_path: Path to test image
    """
    print(f"Testing card detection with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"request_id": "test_123"}
        
        response = requests.post(
            "http://localhost:8000/ocr/cards/detect",
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}\n")
    
    # Validate response structure
    if "request_id" not in result:
        print("Error: Missing request_id in response")
        return False
    
    if "cards" not in result:
        print("Error: Missing cards in response")
        return False
    
    cards = result["cards"]
    print(f"Detected {len(cards)} card(s)")
    
    for card in cards:
        print(f"  Card {card['card_index']}: {card['bounds']}")
    
    return response.status_code in [200, 422]


def main():
    """Run all tests."""
    print("=" * 60)
    print("Team Card Detection Service - Test Suite")
    print("=" * 60 + "\n")
    
    # Test basic endpoints
    if not test_health():
        print("Health check failed!")
        sys.exit(1)
    
    if not test_root():
        print("Root endpoint test failed!")
        sys.exit(1)
    
    # Test card detection (requires image)
    test_image = "/home/admin/ocr-testing/test_image.jpg"
    
    if Path(test_image).exists():
        if not test_card_detection(test_image):
            print("Card detection test failed!")
            sys.exit(1)
    else:
        print(f"Note: Test image not found at {test_image}")
        print("To test card detection, provide a scoreboard screenshot at that path")
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
