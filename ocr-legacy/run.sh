#!/bin/bash

# Team Card Detection Service - Startup Script

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Team Card Detection Service${NC}"
echo "=============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --only-binary :all: -r requirements.txt
fi

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
echo -e "${GREEN}Starting service on http://0.0.0.0:8000${NC}"
echo -e "${GREEN}Auto-reload enabled - changes to code will reload automatically${NC}"
echo "Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
