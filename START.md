# Team Card Detection Service - Start Here

## 🚀 Quick Start (2 minutes)

### 1. Configure AWS
```bash
aws configure
# Enter your AWS credentials and set region to us-east-2
```

### 2. Start Backend
```bash
cd /home/admin/ocr-testing
./run.sh
```

You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Access Frontend

**Option A: Swagger UI (Interactive)**
```
http://18.223.213.127:8000/docs
```

**Option B: Custom HTML Frontend**
```bash
# Serve the frontend on port 8001
python -m http.server 8001
# Then open: http://18.223.213.127:8001/frontend_integration_example.html
```

**Option C: API Documentation**
```
http://18.223.213.127:8000/redoc
```

## 📁 Project Structure

```
/home/admin/ocr-testing/
├── main.py                          # FastAPI backend
├── run.sh                           # Startup script
├── requirements.txt                 # Dependencies
├── favicon.ico                      # Frontend icon
├── frontend_integration_example.html # Web UI
├── test_service.py                  # Tests
├── services/
│   ├── __init__.py
│   └── card_detector.py            # Detection logic
├── prompts/
│   ├── system_prompt.txt
│   ├── user_prompt.txt
│   └── retry_prompt.txt
└── docs/                            # Documentation
    ├── QUICKSTART.md
    ├── README.md
    ├── SETUP.md
    ├── BEDROCK_SETUP.md
    ├── FRONTEND_ACCESS.md
    ├── IMPROVEMENTS.md
    └── ...
```

## 🎯 What It Does

Detects team card bounding boxes from Free Fire scoreboard screenshots using Claude 3.5 Sonnet via AWS Bedrock.

**Input**: Free Fire scoreboard screenshot
**Output**: JSON with card bounding boxes

```json
{
  "request_id": "uuid",
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

## 🔧 Configuration

### AWS Credentials
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-2"
```

### Feature Flag
```bash
# Disable detection (optional)
export OCR_TEAM_CARD_DETECTION_ENABLED=false
```

## 📊 Testing

### Using Swagger UI
1. Open http://localhost:8000/docs
2. Click POST /ocr/cards/detect
3. Click "Try it out"
4. Upload an image
5. Click "Execute"

### Using curl
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@scoreboard.jpg"
```

## 📚 Documentation

See `docs/` directory:
- **QUICKSTART.md** - 2-minute setup
- **README.md** - Complete API reference
- **BEDROCK_SETUP.md** - AWS configuration
- **FRONTEND_ACCESS.md** - Frontend guide
- **IMPROVEMENTS.md** - What was improved

## ✨ Features

✅ Vision-based card detection (Claude 3.5 Sonnet)
✅ AWS Bedrock integration
✅ Strict JSON validation with retry
✅ Comprehensive error handling
✅ Structured logging & observability
✅ Feature flags for control
✅ CORS-enabled for frontend
✅ Interactive Swagger UI
✅ Custom HTML frontend

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check port 8000 is free
lsof -i :8000
```

### Frontend CORS error
- Backend must be running on http://localhost:8000
- Frontend can be on any port
- CORS is enabled for all origins

### Favicon 404
- Favicon is served from http://localhost:8000/favicon.ico
- File exists at `/home/admin/ocr-testing/favicon.ico`

## 🚀 Next Steps

1. Start backend: `./run.sh`
2. Open frontend: `http://localhost:8000/docs`
3. Upload a Free Fire scoreboard screenshot
4. See detected cards with bounding boxes

---

**Status**: ✅ Production Ready
**Backend**: AWS Bedrock (Claude 3.5 Sonnet)
**Frontend**: Swagger UI + Custom HTML
