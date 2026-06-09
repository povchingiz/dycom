# FaceSim Demo - Deployment Guide

Quick start guide for deploying the FaceSim web demo on a GPU server.

---

## Quick Summary

- **Upload:** DICOM file (~100-200MB)
- **Process:** 10-15 minutes on GPU (vs 3-4 hours on CPU)
- **Download:** ZIP with ~58 STL files (~50MB)
- **Cost:** ~$0.15 per demo on RunPod RTX 4090

---

## Step 1: Choose a GPU Server

### Recommended: RunPod (easiest, cheapest for demos)

**Why:** Cheap RTX 4090, pre-configured PyTorch images, easy deployment

1. Go to [runpod.io](https://runpod.io)
2. Sign up, add $10 credit
3. Click "Deploy" → "Template" → Search "PyTorch"
4. Choose: **PyTorch 2.1+ Python 3.12** (or latest)
5. GPU: **RTX 4090** (~$0.40/hr)
6. Storage: 50GB is enough
7. Deploy

You'll get:
- Public URL: `https://<pod-id>-8000.proxy.runpod.net`
- SSH access (optional)
- Jupyter lab (optional)

### Alternatives

| Provider | GPU | Cost/hr | Notes |
|----------|-----|---------|-------|
| Lambda Labs | A10 | ~$0.60 | More enterprise |
| AWS g4dn | T4 | ~$0.53 | Need AWS account |
| Vast.ai | RTX 4090 | ~$0.35 | Cheapest, less reliable |

---

## Step 2: Install Dependencies

SSH into your server or use Jupyter terminal:

```bash
# Clone your repo
git clone <your-repo-url>
cd dycom

# Create Python 3.12 environment
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install dependencies
pip install -r requirements.txt

# This installs:
# - TotalSegmentator (AI segmentation)
# - FastAPI, uvicorn (web server)
# - All medical imaging libs
```

---

## Step 3: Configure Password

```bash
# Copy example env file
cp .env.example .env

# Edit password (IMPORTANT - change default!)
nano .env

# Set: DEMO_PASSWORD=your_secure_password_here
```

---

## Step 4: Run the Server

```bash
# Make script executable
chmod +x run_server.sh

# Run
./run_server.sh
```

Server starts on `http://0.0.0.0:8000`

### On RunPod:

Your server is accessible at:
```
https://<your-pod-id>-8000.proxy.runpod.net
```

Share this URL + password with your demo audience.

---

## Step 5: Test the Demo

1. Open the URL in browser
2. Enter password
3. Upload a test DICOM file (use `data/anon/patient_anon.dcm`)
4. Wait 10-15 minutes
5. Download results ZIP

---

## Cost Management

**RTX 4090 @ $0.40/hr:**
- 1 demo = ~15 min processing = **$0.10**
- 10 demos/day = **$1/day** = **$30/month**

**Tips:**
- Pause pod between demos (don't terminate - keeps storage)
- Paused pod = $0.01/hr storage only
- Resume instantly when needed

---

## Troubleshooting

### "CUDA out of memory"
TotalSegmentator needs ~6GB VRAM. RTX 4090 has 24GB - shouldn't happen. Try:
```bash
# Force CPU fallback (slower but works)
# Edit .env and set:
SEGMENTATION_DEVICE=cpu
```

### "Import error: TotalSegmentator"
```bash
pip uninstall TotalSegmentator
pip install TotalSegmentator>=2.2.0
```

### Server won't start
```bash
# Check port is free
lsof -i :8000

# Kill if needed
kill -9 <PID>
```

### Upload fails (file too large)
Default limit: 500MB. CBCT scans are ~100-200MB. If needed, edit `server/main.py`:
```python
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB
```

---

## Auto-Cleanup

Sessions auto-delete after 7 days. To manually cleanup:

```bash
# Delete all sessions
rm -rf server/sessions/*

# Or delete specific session
rm -rf server/sessions/<session_id>
```

---

## Production Considerations (Future)

For production use (not just demos):

- [ ] HTTPS (use Cloudflare or Let's Encrypt)
- [ ] Rate limiting (prevent abuse)
- [ ] User accounts (not just one password)
- [ ] Database (track usage, store results longer)
- [ ] Cloud storage (S3 for results, not local disk)
- [ ] Monitoring (Sentry, logging)
- [ ] Queue system (Celery + Redis for multiple users)

---

## Support

Issues? Check:
- Server logs: stdout from `python main.py`
- Browser console: F12 → Console tab
- Session status: `GET /sessions` endpoint

Contact: yertaychingiz@gmail.com
