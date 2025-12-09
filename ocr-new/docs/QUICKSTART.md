# Team Card Detection Service - Quick Start

## 🚀 Get Running in 2 Minutes

### 1. Configure AWS Credentials
```bash
# Configure AWS credentials (if not already done)
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-2"
```

The service uses AWS Bedrock with Claude 3.5 Sonnet global inference profile.

### 2. Start Server (Automatic Setup)
```bash
cd /home/admin/ocr-testing
./run.sh
```

Or manually:
```bash
cd /home/admin/ocr-testing
python3 -m venv venv
source venv/bin/activate
pip install --only-binary :all: -r requirements.txt
python main.py
```

You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 📡 Test the API

### Using curl
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg"
```

### Using Python
```python
import requests

with open("scoreboard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f}
    )
    print(response.json())
```

### Using Browser
Open `frontend_integration_example.html` in your browser and upload an image.

## 📊 Expected Response

```json
{
  "request_id": "uuid-here",
  "cards": [
    {
      "card_index": 1,
      "bounds": {
        "x1": 120,
        "y1": 210,
        "x2": 880,
        "y2": 460
      }
    }
  ]
}
```

## 📁 Project Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application |
| `services/card_detector.py` | Detection logic |
| `prompts/*.txt` | Claude prompts |
| `requirements.txt` | Dependencies |
| `README.md` | Full API docs |
| `USAGE.md` | Usage examples |
| `IMPLEMENTATION_GUIDE.md` | Deep dive |
| `test_service.py` | Test suite |
| `frontend_integration_example.html` | Web UI |

## 🔧 What It Does

✅ Accepts Free Fire scoreboard screenshots
✅ Uses Claude 3.5 Sonnet Vision to detect team cards
✅ Returns bounding box coordinates for each card
✅ Validates JSON responses with automatic retry
✅ CORS-enabled for frontend integration

## 📝 Endpoint Details

**POST /ocr/cards/detect**

**Input:**
- `file` (required): Image file
- `request_id` (optional): Request identifier
- `game_metadata` (optional): Game metadata

**Output:**
- `request_id`: Echo of input request_id
- `cards`: Array of detected cards with bounds
- `error`: Error message if applicable

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Run `python main.py` |
| API key error | Set `ANTHROPIC_API_KEY` |
| Invalid image | Provide valid image file |
| No cards detected | Check image is Free Fire scoreboard |

## 📚 Learn More

- **Full API Docs**: See `README.md`
- **Usage Examples**: See `USAGE.md`
- **Implementation Details**: See `IMPLEMENTATION_GUIDE.md`
- **Interactive API Docs**: http://localhost:8000/docs

## 🎯 Next Steps

1. Test with a Free Fire scoreboard screenshot
2. Integrate with your frontend using the example HTML
3. Customize prompts if needed (in `prompts/` folder)
4. Deploy to production when ready

---

**Need help?** Check the documentation files or review the code comments in `services/card_detector.py`.
