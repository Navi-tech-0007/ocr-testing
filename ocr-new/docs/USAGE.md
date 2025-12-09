# Team Card Detection Service - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
cd /home/admin/ocr-testing
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Start the Service
```bash
python main.py
```

The service will be available at `http://localhost:8000`

## API Usage Examples

### Using curl

```bash
# Basic request with image file
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg"

# With request ID
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg" \
  -F "request_id=my_request_123"

# With all optional fields
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg" \
  -F "request_id=my_request_123" \
  -F "game_metadata={\"match_id\": \"abc123\"}"
```

### Using Python

```python
import requests
import json

# Simple request
with open("scoreboard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f}
    )
    print(response.json())

# With request ID
with open("scoreboard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f},
        data={"request_id": "test_123"}
    )
    result = response.json()
    print(f"Request ID: {result['request_id']}")
    print(f"Cards detected: {len(result['cards'])}")
    for card in result['cards']:
        print(f"  Card {card['card_index']}: {card['bounds']}")
```

### Using JavaScript/Fetch

```javascript
// Fetch API example
const formData = new FormData();
const imageInput = document.getElementById('image-input');
formData.append('file', imageInput.files[0]);
formData.append('request_id', 'web_request_' + Date.now());

fetch('http://localhost:8000/ocr/cards/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Request ID:', data.request_id);
  console.log('Cards detected:', data.cards.length);
  data.cards.forEach(card => {
    console.log(`Card ${card.card_index}:`, card.bounds);
  });
})
.catch(error => console.error('Error:', error));
```

## Response Examples

### Successful Detection (200)
```json
{
  "request_id": "test_123",
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

### No Cards Detected (200)
```json
{
  "request_id": "test_123",
  "cards": [],
  "error": "NO_CARDS_DETECTED"
}
```

### Invalid JSON Response (422)
```json
{
  "request_id": "test_123",
  "error": "Failed to parse JSON after retry: Invalid JSON response from Claude",
  "cards": []
}
```

### Invalid Image (400)
```json
{
  "detail": "File must be an image (image/*)"
}
```

## Understanding the Response

Each detected card contains:

- **card_index** (integer): Sequential order from top to bottom (1 = topmost)
- **bounds** (object): Bounding box coordinates
  - **x1**: Left edge (pixels from left)
  - **y1**: Top edge (pixels from top)
  - **x2**: Right edge (pixels from left)
  - **y2**: Bottom edge (pixels from top)

Example interpretation:
```
Card 1 bounds: x1=120, y1=210, x2=880, y2=460
- Located 120 pixels from left edge
- Located 210 pixels from top edge
- Width: 880 - 120 = 760 pixels
- Height: 460 - 210 = 250 pixels
```

## Testing the Service

### Run Test Suite
```bash
python test_service.py
```

This will test:
- Health check endpoint
- Root endpoint
- Card detection (if test image exists)

### Manual Testing with Postman

1. Create a new POST request to `http://localhost:8000/ocr/cards/detect`
2. Go to "Body" tab
3. Select "form-data"
4. Add key "file" (type: File) and select your scoreboard image
5. Add key "request_id" (type: Text) with value like "postman_test_1"
6. Click "Send"

## Troubleshooting

### "Invalid image" Error
- Ensure the file is a valid image format (JPEG, PNG, etc.)
- Check file is not corrupted
- Verify file size is reasonable

### "Failed to parse JSON" Error
- The Claude model returned invalid JSON
- Service automatically retries once
- If still failing, check if image is a valid Free Fire scoreboard
- May indicate model confusion with unusual layouts

### "NO_CARDS_DETECTED" Error
- No team cards were found in the image
- Verify image is a Free Fire scoreboard screenshot
- Check if cards are visible and not obscured
- Try with a different screenshot

### Connection Refused
- Ensure service is running: `python main.py`
- Check port 8000 is not in use: `lsof -i :8000`
- Verify ANTHROPIC_API_KEY is set

### API Key Error
- Set environment variable: `export ANTHROPIC_API_KEY="your-key"`
- Verify key is valid and has API access
- Check key doesn't have leading/trailing spaces

## Performance Notes

- First request may take 2-3 seconds (Claude API latency)
- Subsequent requests typically 1-2 seconds
- Image size affects processing time (larger images = slower)
- Service handles one request at a time (single detector instance)

## Integration with Frontend

The service is CORS-enabled and ready for frontend integration:

```javascript
// Frontend can call the service directly
const detectCards = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/ocr/cards/detect', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};
```

## Advanced Configuration

### Changing Model Parameters

Edit `/home/admin/ocr-testing/services/card_detector.py`:

```python
def _call_claude_vision(self, image_base64: str, prompt_text: str, is_retry: bool = False) -> str:
    message = self.client.messages.create(
        model=self.model,
        max_tokens=1500,      # Adjust here
        temperature=0,         # Adjust here (0-1)
        top_p=1,              # Adjust here
        top_k=1,              # Adjust here
        # ... rest of config
    )
```

### Customizing Prompts

Edit the prompt files in `/home/admin/ocr-testing/prompts/`:
- `system_prompt.txt`: Claude's role and behavior
- `user_prompt.txt`: Main detection task
- `retry_prompt.txt`: Fallback for invalid JSON

## Support

For issues or questions:
1. Check the README.md for architecture details
2. Review error messages in service logs
3. Test with `test_service.py`
4. Verify image quality and format
