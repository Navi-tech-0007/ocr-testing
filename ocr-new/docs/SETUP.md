# Team Card Detection Service - Setup Guide

## ✅ Installation Complete

Your Team Card Detection Service is ready to use!

## 📋 What Was Installed

- ✅ FastAPI 0.124.0 - Web framework
- ✅ Uvicorn 0.38.0 - ASGI server
- ✅ Boto3 1.42.5 - AWS SDK for Bedrock
- ✅ Pillow 12.0.0 - Image processing
- ✅ Pydantic 2.12.5 - Data validation
- ✅ Python-multipart 0.0.20 - Form handling

All dependencies installed in virtual environment: `/home/admin/ocr-testing/venv`

**Backend**: AWS Bedrock with Claude 3.5 Sonnet global inference profile

## 🚀 Quick Start

### Option 1: Using the Startup Script (Recommended)

```bash
cd /home/admin/ocr-testing
./run.sh
```

The script will:
1. Create virtual environment if needed
2. Activate it
3. Install dependencies if needed
4. Start the service

### Option 2: Manual Startup

```bash
cd /home/admin/ocr-testing
source venv/bin/activate
python main.py
```

## 🔑 AWS Credentials Setup

The service uses AWS Bedrock. Configure your credentials:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-2"

# Option 3: Add to shell profile (~/.bashrc, ~/.zshrc)
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-2"
```

Required IAM permissions:
- `bedrock:InvokeModel` on `arn:aws:bedrock:us-east-2:*:inference-profile/global.anthropic.claude-3-5-sonnet-*`

## 📡 Verify Installation

### 1. Check Service is Running
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### 2. Test Card Detection
```bash
curl -X POST http://localhost:8000/ocr/cards/detect \
  -F "file=@/path/to/scoreboard.jpg"
```

### 3. View API Documentation
Open in browser: http://localhost:8000/docs

## 📁 Project Structure

```
/home/admin/ocr-testing/
├── 📄 Documentation
│   ├── QUICKSTART.md              ← Start here
│   ├── README.md                  ← Full API docs
│   ├── USAGE.md                   ← Usage examples
│   ├── IMPLEMENTATION_GUIDE.md    ← Deep dive
│   ├── SERVICE_SUMMARY.md         ← Feature overview
│   └── SETUP.md                   ← This file
│
├── 🔧 Service
│   ├── main.py                    ← FastAPI app
│   ├── run.sh                     ← Startup script
│   ├── requirements.txt           ← Dependencies
│   └── services/
│       ├── __init__.py
│       └── card_detector.py       ← Core logic
│
├── 📝 Configuration
│   └── prompts/
│       ├── system_prompt.txt
│       ├── user_prompt.txt
│       └── retry_prompt.txt
│
├── 🧪 Testing
│   ├── test_service.py
│   └── frontend_integration_example.html
│
└── 🐍 Virtual Environment
    └── venv/                      ← Python packages
```

## 🧪 Run Tests

```bash
cd /home/admin/ocr-testing
source venv/bin/activate
python test_service.py
```

This will test:
- Health check endpoint
- Root endpoint
- Card detection (if test image exists)

## 🌐 Frontend Integration

Open the web UI in your browser:
```bash
open /home/admin/ocr-testing/frontend_integration_example.html
```

Or use the Python example:
```python
import requests

with open("scoreboard.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/cards/detect",
        files={"file": f}
    )
    print(response.json())
```

## 🔧 Troubleshooting

### Service Won't Start

**Error: "Address already in use"**
```bash
# Kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or use different port - edit main.py:
# uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Error: "ANTHROPIC_API_KEY not found"**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify it's set
echo $ANTHROPIC_API_KEY
```

### Virtual Environment Issues

**Error: "venv not activated"**
```bash
cd /home/admin/ocr-testing
source venv/bin/activate
```

**Error: "ModuleNotFoundError"**
```bash
# Reinstall dependencies
source venv/bin/activate
pip install --only-binary :all: -r requirements.txt
```

### API Issues

**Error: "Invalid image"**
- Ensure file is valid image (JPEG, PNG, etc.)
- Check file is not corrupted
- Verify file is not empty

**Error: "Failed to parse JSON"**
- Check image is Free Fire scoreboard
- Try with different screenshot
- Verify image quality

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 2-minute setup
- **[README.md](README.md)** - Complete API reference
- **[USAGE.md](USAGE.md)** - Usage examples
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Architecture details
- **[SERVICE_SUMMARY.md](SERVICE_SUMMARY.md)** - Feature overview

## 🎯 Next Steps

1. **Test the service** with a Free Fire scoreboard screenshot
2. **Integrate with frontend** using the provided HTML example
3. **Customize prompts** if needed (in `prompts/` folder)
4. **Deploy** to production when ready

## 📞 Support

For issues or questions:
1. Check the relevant documentation file
2. Review error messages in service logs
3. Run `test_service.py` to verify setup
4. Check API docs at http://localhost:8000/docs

## ✨ You're All Set!

Your Team Card Detection Service is ready to use. Start with:

```bash
cd /home/admin/ocr-testing
export ANTHROPIC_API_KEY="your-key"
./run.sh
```

Then visit: http://localhost:8000/docs

---

**Installation Date**: December 8, 2025
**Python Version**: 3.13
**Virtual Environment**: `/home/admin/ocr-testing/venv`
**Status**: ✅ Ready to Use
