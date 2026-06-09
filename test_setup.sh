#!/bin/bash
# Test script to verify web demo setup

set -e

echo "🧪 Testing FaceSim Web Demo Setup..."
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version | grep "3.12" || {
    echo "⚠️  Warning: Python 3.12 required, found:"
    python --version
}

# Check required packages
echo "✓ Checking required packages..."
python -c "import fastapi" 2>/dev/null && echo "  ✓ fastapi installed" || echo "  ✗ fastapi MISSING"
python -c "import uvicorn" 2>/dev/null && echo "  ✓ uvicorn installed" || echo "  ✗ uvicorn MISSING"
python -c "import TotalSegmentator" 2>/dev/null && echo "  ✓ TotalSegmentator installed" || echo "  ✗ TotalSegmentator MISSING"

# Check file structure
echo "✓ Checking file structure..."
[ -f "server/main.py" ] && echo "  ✓ server/main.py exists" || echo "  ✗ server/main.py MISSING"
[ -f "server/auth.py" ] && echo "  ✓ server/auth.py exists" || echo "  ✗ server/auth.py MISSING"
[ -f "server/pipeline.py" ] && echo "  ✓ server/pipeline.py exists" || echo "  ✗ server/pipeline.py MISSING"
[ -f "server/static/index.html" ] && echo "  ✓ server/static/index.html exists" || echo "  ✗ server/static/index.html MISSING"
[ -f "server/static/app.js" ] && echo "  ✓ server/static/app.js exists" || echo "  ✗ server/static/app.js MISSING"
[ -f "server/static/styles.css" ] && echo "  ✓ server/static/styles.css exists" || echo "  ✗ server/static/styles.css MISSING"

# Check .env
echo "✓ Checking configuration..."
if [ -f ".env" ]; then
    echo "  ✓ .env file exists"
    if grep -q "DEMO_PASSWORD=" .env; then
        echo "  ✓ DEMO_PASSWORD configured"
    else
        echo "  ✗ DEMO_PASSWORD not set in .env"
    fi
else
    echo "  ⚠️  .env not found (copy .env.example .env)"
fi

# Check scripts
echo "✓ Checking segmentation scripts..."
[ -f "scripts/dcm_to_nifti.py" ] && echo "  ✓ dcm_to_nifti.py exists" || echo "  ✗ dcm_to_nifti.py MISSING"
[ -f "scripts/run_teeth_seg.py" ] && echo "  ✓ run_teeth_seg.py exists" || echo "  ✗ run_teeth_seg.py MISSING"
[ -f "scripts/segment_soft_tissue.py" ] && echo "  ✓ segment_soft_tissue.py exists" || echo "  ✗ segment_soft_tissue.py MISSING"
[ -f "scripts/masks_to_stl.py" ] && echo "  ✓ masks_to_stl.py exists" || echo "  ✗ masks_to_stl.py MISSING"

echo ""
echo "✅ Setup check complete!"
echo ""
echo "Next steps:"
echo "1. If .env is missing: cp .env.example .env && nano .env"
echo "2. Start server: ./run_server.sh"
echo "3. Open: http://localhost:8000"
