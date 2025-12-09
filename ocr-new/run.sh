#!/bin/bash

# OCR Pipeline Service (ocr-new) - Startup Script
# Step 2 + 2.5: Static Geometry (no Claude)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}OCR Pipeline Service (ocr-new)${NC}"
echo "================================"
echo "Step 2 + 2.5: Static Geometry"
echo ""

# Note: Using system Python with pre-installed dependencies
# (venv not needed due to system-managed environment)

# Check for AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ -z "$AWS_PROFILE" ]; then
    echo -e "${YELLOW}Warning: AWS credentials not detected${NC}"
    echo "Please configure AWS credentials:"
    echo "  aws configure"
    echo "Or set environment variables:"
    echo "  export AWS_ACCESS_KEY_ID='your-key'"
    echo "  export AWS_SECRET_ACCESS_KEY='your-secret'"
    echo "  export AWS_DEFAULT_REGION='us-east-2'"
    echo ""
fi

# Start the service
echo -e "${GREEN}Starting OCR Pipeline Service on http://0.0.0.0:8000${NC}"
echo -e "${GREEN}Services:${NC}"
echo "  - Step 1: Card Detection (Claude Vision)"
echo "  - Step 2: Player Slots (Static Geometry - no Claude)"
echo "  - Step 2.5: Kill-Box Refinement (Static Geometry - no Claude)"
echo "  - Step 3: Kill Extraction (Tesseract + Claude fallback)"
echo "  - Step 4: Name Extraction (Claude Vision)"
echo "  - Step 5: Final Assembly"
echo ""
echo -e "${GREEN}Auto-reload enabled - changes to code will reload automatically${NC}"
echo "Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
