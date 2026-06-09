# FaceSim Web Demo - Quick Reference

## 🚀 Quick Start (Local Testing)

```bash
cd /Users/yerta/Desktop/dycom

# Activate Python 3.12 environment
source .venv312/bin/activate

# Copy env file and set password
cp .env.example .env
nano .env  # Edit DEMO_PASSWORD

# Run server
./run_server.sh
```

Access at: **http://localhost:8000**

---

## 🌐 Deploy on GPU Server (RunPod)

### 1. Create Pod
- Go to runpod.io
- Deploy: **PyTorch 2.1+ Python 3.12** template
- GPU: **RTX 4090** (~$0.40/hr)
- Storage: 50GB

### 2. Install & Run
```bash
git clone <your-repo>
cd dycom
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env  # Set password

./run_server.sh
```

### 3. Access
URL: `https://<pod-id>-8000.proxy.runpod.net`

---

## 📁 What Was Created

```
dycom/
├── server/
│   ├── main.py              # FastAPI backend
│   ├── auth.py              # Password authentication
│   ├── pipeline.py          # Segmentation orchestrator
│   └── static/
│       ├── index.html       # Web UI (login, upload, progress, download)
│       ├── app.js           # Frontend logic (EN/RU/KK, dark/light)
│       └── styles.css       # Minimal clean styling
├── scripts/                 # Updated with progress callbacks
├── .env.example             # Password config template
├── run_server.sh            # Startup script
├── DEPLOYMENT.md            # Full deployment guide
└── requirements.txt         # Updated with web deps
```

---

## ✨ Features

- 🔐 **Password protection** (configurable in .env)
- 📤 **Drag & drop upload** (DICOM files up to 500MB)
- ⏱️ **Live progress** (5 steps with circular progress indicator)
- 📥 **Download ZIP** (all ~58 STL files)
- 🌙 **Dark/Light theme** (auto-saved preference)
- 🌐 **3 languages** (English, Russian, Kazakh)
- 🗑️ **Session management** (delete after download, auto-cleanup after 7 days)
- 📱 **Responsive design** (works on mobile)

---

## ⚡ Performance

| Hardware | Time | Cost |
|----------|------|------|
| MacBook (CPU) | 3-4 hours | Free |
| RTX 4090 (GPU) | 10-15 min | ~$0.10 |

---

## 🎯 Demo Flow

1. **Login** → Enter password
2. **Upload** → Drag DICOM file
3. **Process** → 5 steps (10-15 min on GPU):
   - Converting DICOM to NIfTI
   - Segmenting teeth and jawbones
   - Segmenting soft tissue
   - Generating 3D meshes (STL)
   - Preparing download package
4. **Download** → ZIP with ~58 STL files
5. **Delete** (optional) → Remove session

---

## 🛠️ Troubleshooting

**Server won't start:**
```bash
# Check Python version (must be 3.12)
python --version

# Reinstall dependencies
pip install -r requirements.txt
```

**Upload fails:**
- Max file size: 500MB
- Accepted format: .dcm only

**Segmentation fails:**
- Check GPU memory (needs ~6GB VRAM)
- Fallback to CPU: edit `server/pipeline.py`, change `device="cuda"` to `device="cpu"`

---

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_PASSWORD` | `nine9une@)@^` | Web access password |

---

## 🔒 Security Notes

For production use:
- [ ] Change default password
- [ ] Enable HTTPS (RunPod proxy provides this)
- [ ] Add rate limiting
- [ ] Implement user accounts (not just one password)
- [ ] Set up monitoring/logging

---

## 📞 Contact

Yerta — yertaychingiz@gmail.com
