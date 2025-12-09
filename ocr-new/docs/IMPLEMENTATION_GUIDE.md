# Team Card Detection Service - Implementation Guide

## Overview

This is a complete backend service for detecting team card bounding boxes from Free Fire scoreboard screenshots. It uses Claude 3.5 Sonnet Vision to analyze images and return precise bounding box coordinates.

## What This Service Does

✅ **Accepts**: Free Fire scoreboard screenshots (any image format)
✅ **Detects**: Team card bounding boxes (each card = 2×2 grid of players)
✅ **Returns**: JSON with card indices and pixel coordinates
✅ **Validates**: Strict schema enforcement with automatic retry
✅ **Integrates**: CORS-enabled for frontend use

## Project Structure

```
/home/admin/ocr-testing/
├── main.py                          # FastAPI application
├── requirements.txt                 # Python dependencies
├── README.md                        # API documentation
├── USAGE.md                         # Usage examples
├── IMPLEMENTATION_GUIDE.md          # This file
├── test_service.py                  # Test suite
├── frontend_integration_example.html # Frontend UI example
├── services/
│   ├── __init__.py
│   └── card_detector.py            # Core detection logic
└── prompts/
    ├── system_prompt.txt           # Claude system prompt
    ├── user_prompt.txt             # Card detection task
    └── retry_prompt.txt            # Invalid JSON recovery
```

## Installation & Setup

### Step 1: Install Dependencies
```bash
cd /home/admin/ocr-testing
pip install -r requirements.txt
```

**Dependencies:**
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `python-multipart==0.0.6` - Multipart form handling
- `anthropic==0.25.1` - Claude API client
- `pillow==10.1.0` - Image processing
- `pydantic==2.5.0` - Data validation

### Step 2: Set API Key
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

Get your API key from: https://console.anthropic.com/

### Step 3: Start the Service
```bash
python main.py
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

## API Specification

### Endpoint: POST /ocr/cards/detect

**Purpose**: Detect team card bounding boxes from a scoreboard screenshot

**Request Format**:
```
Content-Type: multipart/form-data

Parameters:
- file (required): Image file (image/*)
- request_id (optional): Request identifier string
- game_metadata (optional): Game metadata JSON (ignored)
```

**Response Format (Success - 200)**:
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

**Response Format (No Cards - 200)**:
```json
{
  "request_id": "req_123",
  "cards": [],
  "error": "NO_CARDS_DETECTED"
}
```

**Response Format (Invalid JSON - 422)**:
```json
{
  "request_id": "req_123",
  "error": "Failed to parse JSON after retry: ...",
  "cards": []
}
```

**Response Format (Invalid Image - 400)**:
```json
{
  "detail": "File must be an image (image/*)"
}
```

## Service Architecture

### 1. Image Validation
```python
# Validates image format and dimensions
- Checks MIME type (must be image/*)
- Opens image with PIL
- Returns width and height
- Raises error if invalid
```

### 2. Base64 Encoding
```python
# Converts image bytes to base64 for Claude API
- Uses standard base64 encoding
- Passes to Claude as image source
```

### 3. Claude Vision Call
```python
# Sends image + prompts to Claude 3.5 Sonnet
- Model: claude-3-5-sonnet-20241022
- Max tokens: 1500
- Temperature: 0 (deterministic)
- System prompt: Enforces JSON-only output
- User prompt: Specifies card detection task
```

### 4. JSON Parsing
```python
# Extracts JSON from response
- Attempts direct JSON parsing
- Falls back to finding JSON object in text
- Raises error if both fail
```

### 5. Schema Validation
```python
# Validates response structure
- Checks "cards" key exists
- Validates each card has required fields
- Validates card_index is integer
- Validates bounds are integers
- Validates bounds are within image dimensions
- Validates x1 < x2 and y1 < y2
```

### 6. Sorting & Return
```python
# Sorts cards and returns response
- Sorts by card_index ascending
- Removes approx_visual_rank from response
- Returns only card_index and bounds
- Adds request_id to response
```

## Code Walkthrough

### Main Application (main.py)

```python
# Initialize FastAPI app
app = FastAPI(...)

# Enable CORS for frontend
app.add_middleware(CORSMiddleware, ...)

# Initialize detector
detector = TeamCardDetector()

# Define endpoint
@app.post("/ocr/cards/detect")
async def detect_team_cards(file, request_id, game_metadata):
    # Validate image
    # Read bytes
    # Call detector
    # Return response
```

### Card Detector Service (services/card_detector.py)

```python
class TeamCardDetector:
    def __init__(self):
        # Initialize Anthropic client
        # Load prompts from files
    
    def detect_cards(self, image_bytes):
        # Validate image
        # Convert to base64
        # Call Claude Vision
        # Parse JSON
        # Validate schema
        # Sort and return
```

## Prompts Used

### System Prompt
Instructs Claude to:
- Act as a strict vision-analysis API
- Output valid JSON only
- Use integer coordinates
- Never include explanations
- Return null for uncertain values

### User Prompt
Specifies:
- Task: Detect team cards in Free Fire scoreboard
- Card structure: 2×2 grid of players
- Output format: JSON with card_index, approx_visual_rank, bounds
- Rules: Include partial cards, no names/kills, estimate if uncertain

### Retry Prompt
Used if JSON parsing fails:
- Instructs Claude to return strict JSON
- No comments, text, or markdown
- Same structure as before

## Error Handling

### Invalid Image (400)
```
Cause: File is not an image or is corrupted
Fix: Provide valid image file (JPEG, PNG, etc.)
```

### Invalid JSON (422)
```
Cause: Claude returned malformed JSON
Fix: Service retries once with fallback prompt
If still fails: Check if image is valid Free Fire scoreboard
```

### No Cards Detected (200)
```
Cause: No team cards found in image
Fix: Verify image is Free Fire scoreboard screenshot
Try different screenshot if cards are obscured
```

### Connection Error
```
Cause: Service not running or API key invalid
Fix: 
1. Start service: python main.py
2. Set API key: export ANTHROPIC_API_KEY="..."
3. Check port 8000 is available
```

## Testing

### Run Test Suite
```bash
python test_service.py
```

Tests:
- Health check endpoint
- Root endpoint
- Card detection (if test image exists)

### Manual Testing with curl
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg" \
  -F "request_id=test_123"
```

### Manual Testing with Python
```python
import requests

with open("scoreboard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f},
        data={"request_id": "test_123"}
    )
    print(response.json())
```

### Frontend Testing
Open `frontend_integration_example.html` in browser:
- Drag and drop image
- Click "Detect Cards"
- View results with bounding boxes

## Integration Examples

### JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('request_id', 'web_' + Date.now());

const response = await fetch('http://localhost:8000/ocr/cards/detect', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data.cards);
```

### Python/Requests
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f},
        data={"request_id": "py_test"}
    )
    result = response.json()
    for card in result['cards']:
        print(f"Card {card['card_index']}: {card['bounds']}")
```

### cURL
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@scoreboard.jpg" \
  -F "request_id=curl_test"
```

## Performance Characteristics

- **First Request**: 2-3 seconds (Claude API latency)
- **Subsequent Requests**: 1-2 seconds
- **Image Size Impact**: Larger images = slower processing
- **Concurrency**: Single detector instance (sequential processing)

## Customization

### Change Claude Model
Edit `services/card_detector.py`:
```python
self.model = "claude-3-5-sonnet-20241022"  # Change this
```

### Adjust Token Limits
Edit `services/card_detector.py`:
```python
max_tokens=1500,  # Increase for longer responses
```

### Modify Prompts
Edit files in `prompts/`:
- `system_prompt.txt` - Change Claude's role
- `user_prompt.txt` - Change detection task
- `retry_prompt.txt` - Change retry behavior

### Change Port
Edit `main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port here
```

## Troubleshooting

### Service Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Try different port
# Edit main.py and change port number
```

### API Key Error
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY="sk-ant-..."

# Check key is valid at https://console.anthropic.com/
```

### Image Validation Error
```
Ensure image:
- Is valid format (JPEG, PNG, GIF, etc.)
- Is not corrupted
- Has reasonable dimensions
- Is not empty file
```

### JSON Parsing Error
```
Check:
- Image is valid Free Fire scoreboard
- Cards are clearly visible
- Image quality is good
- Try with different screenshot
```

## Deployment Considerations

### Production Setup
1. Use environment variables for API key
2. Add request logging
3. Implement rate limiting
4. Add database for persistence
5. Use process manager (gunicorn, supervisor)
6. Enable HTTPS
7. Add authentication if needed

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
CMD ["python", "main.py"]
```

### Scaling
- Load balance across multiple instances
- Use shared cache for images
- Implement request queuing
- Monitor API usage and costs

## Support & Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Service Health
```bash
curl http://localhost:8000/health
```

### View API Docs
```
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc # ReDoc
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| 400 Bad Request | Invalid image | Provide valid image file |
| 422 Unprocessable | Invalid JSON | Check image is scoreboard |
| 500 Server Error | API key invalid | Set ANTHROPIC_API_KEY |
| Connection refused | Service not running | Run `python main.py` |
| Timeout | Slow API | Check internet connection |

## Next Steps

1. **Test the service** with sample Free Fire screenshots
2. **Integrate with frontend** using provided HTML example
3. **Monitor performance** and adjust token limits if needed
4. **Add persistence** if you need to store results
5. **Deploy** to production environment

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Pydantic Validation](https://docs.pydantic.dev/)
- [Pillow Image Library](https://python-pillow.org/)
