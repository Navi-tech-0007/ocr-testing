# Team Card Detection Service

A FastAPI backend service for detecting team card bounding boxes from Free Fire scoreboard screenshots using Claude 3.5 Sonnet Vision.

## Features

- **Vision-based Detection**: Uses Claude 3.5 Sonnet via AWS Bedrock to analyze scoreboard images
- **Strict JSON Validation**: Enforces schema validation with retry logic
- **Bounding Box Extraction**: Returns pixel-perfect coordinates for each team card
- **Error Handling**: Comprehensive error handling with meaningful messages
- **CORS Enabled**: Ready for frontend integration
- **AWS Bedrock Integration**: Uses global inference profile for optimal performance

## Project Structure

```
/home/admin/ocr-testing/
├── main.py                 # FastAPI application and endpoints
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── services/
│   ├── __init__.py
│   └── card_detector.py   # Core detection service
└── prompts/
    ├── system_prompt.txt  # Claude system prompt
    ├── user_prompt.txt    # Team card detection prompt
    └── retry_prompt.txt   # Fallback prompt for invalid JSON
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS Credentials:

The service uses AWS Bedrock with the global Sonnet inference profile. Ensure your AWS credentials are configured:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-2"

# Option 3: IAM Role (if running on EC2)
# Attach IAM role with bedrock:InvokeModel permissions
```

Required IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-2:*:inference-profile/global.anthropic.claude-3-5-sonnet-*"
    }
  ]
}
```

## Running the Service

Start the server:
```bash
python main.py
```

The service will start on `http://localhost:8000`

## API Endpoints

### POST /ocr/cards/detect

Detect team card bounding boxes from a scoreboard screenshot.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required): Image file (image/*)
  - `request_id` (optional): Request identifier string
  - `game_metadata` (optional): Game metadata (ignored)

**Response (Success - 200):**
```json
{
  "request_id": "req_123",
  "cards": [
    {
      "card_index": 1,
      "bounds": {
        "x1": 120,
        "y1": 210,
        "x2": 880,
        "y2": 460
      }
    },
    {
      "card_index": 2,
      "bounds": {
        "x1": 120,
        "y1": 470,
        "x2": 880,
        "y2": 720
      }
    }
  ]
}
```

**Response (No Cards - 200):**
```json
{
  "request_id": "req_123",
  "cards": [],
  "error": "NO_CARDS_DETECTED"
}
```

**Response (Invalid JSON - 422):**
```json
{
  "request_id": "req_123",
  "error": "Failed to parse JSON after retry: ...",
  "cards": []
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /

Service info endpoint.

**Response:**
```json
{
  "service": "Team Card Detection Service",
  "version": "1.0.0",
  "endpoint": "POST /ocr/cards/detect",
  "health": "GET /health"
}
```

## Testing

### Using curl:
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg" \
  -F "request_id=test_123"
```

### Using Python:
```python
import requests

with open("scoreboard.jpg", "rb") as f:
    files = {"file": f}
    data = {"request_id": "test_123"}
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files=files,
        data=data
    )
    print(response.json())
```

## Service Details

### Claude Model Configuration
- **Model**: claude-3-5-sonnet-20241022
- **Max Tokens**: 1500
- **Temperature**: 0 (deterministic)
- **Top P**: 1
- **Top K**: 1

### Validation Rules
- Card indices must be sequential integers
- Bounding boxes must have integer coordinates
- Coordinates must be within image dimensions
- x1 < x2 and y1 < y2
- Cards are sorted by card_index before returning

### Error Handling
- Invalid JSON triggers automatic retry with fallback prompt
- Invalid bounding boxes return HTTP 422
- Missing required fields return HTTP 422
- Invalid images return HTTP 400

## Architecture

1. **Image Validation**: Validates image format and dimensions
2. **Base64 Encoding**: Converts image to base64 for Claude API
3. **Claude Vision Call**: Sends image and prompts to Claude 3.5 Sonnet
4. **JSON Parsing**: Extracts JSON from response with fallback extraction
5. **Schema Validation**: Validates cards schema and bounding boxes
6. **Sorting**: Orders cards by card_index
7. **Response Formatting**: Returns cleaned JSON response

## Notes

- The service detects only team card bounding boxes, not individual players
- Each team card contains exactly 4 player slots in a 2×2 grid
- Partially visible cards are included in detection
- The service uses strict JSON validation to ensure data quality
