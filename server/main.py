"""
FaceSim Demo Server - FastAPI Backend
Upload DICOM, process segmentation, download STL results.
"""
import asyncio
import os
import pathlib
import shutil
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from auth import (
    DEMO_PASSWORD,
    Session,
    sessions,
    verify_password,
    create_session,
    get_session,
    cleanup_old_sessions,
)
from pipeline import run_pipeline, PipelineError


# Configuration
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
SESSION_DIR = pathlib.Path(__file__).parent / "sessions"
SESSION_DIR.mkdir(exist_ok=True)

# Cleanup old sessions on startup
removed = cleanup_old_sessions(max_age_hours=24 * 7)
print(f"Cleaned up {removed} old sessions")

# FastAPI app
app = FastAPI(title="FaceSim Demo", version="1.0.0")

# CORS (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(pathlib.Path(__file__).parent / "static")), name="static")


# ============== Models ==============

class LoginRequest(BaseModel):
    password: str


class ProgressUpdate(BaseModel):
    session_id: str
    message: str
    percentage: int


class SessionStatus(BaseModel):
    session_id: str
    status: str
    message: Optional[str] = None
    error: Optional[str] = None
    download_url: Optional[str] = None
    simulation: Optional[dict] = None


# ============== Middleware ==============

@app.middleware("http")
async def check_auth_middleware(request: Request, call_next):
    """Check authentication for protected routes."""
    # Public routes
    if request.url.path in ["/", "/static/index.html", "/health", "/login"]:
        return await call_next(request)
    
    # Check session cookie for protected routes
    session_cookie = request.cookies.get("session_auth")
    if not session_cookie or session_cookie != "authenticated":
        if request.url.path.startswith("/static"):
            return await call_next(request)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    return await call_next(request)


# ============== Routes ==============

@app.get("/")
async def root():
    """Redirect to login page."""
    return FileResponse(SESSION_DIR.parent / "static" / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/login")
async def login(request: LoginRequest):
    """Authenticate with password."""
    if verify_password(request.password):
        response = JSONResponse(content={"success": True})
        response.set_cookie(
            key="session_auth",
            value="authenticated",
            httponly=True,
            max_age=60 * 60 * 24 * 7,  # 7 days
            samesite="lax",
        )
        return response
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )


@app.post("/logout")
async def logout():
    """Logout (clear auth cookie)."""
    response = JSONResponse(content={"success": True})
    response.delete_cookie("session_auth")
    return response


@app.post("/upload")
async def upload_scan(
    file: UploadFile = File(...),
    advance_mm: float = Form(5.0),
    vertical_mm: float = Form(0.0),
    lateral_mm: float = Form(0.0),
    pitch_deg: float = Form(0.0),
    simulate: bool = Form(True),
):
    """
    Загрузка DICOM и запуск обработки.

    Параметры операции (мм / градусы), задаются хирургом:
      advance_mm   — выдвижение нижней челюсти вперёд (отрицательное = назад)
      vertical_mm  — смещение вверх
      lateral_mm   — смещение влево (коррекция асимметрии)
      pitch_deg    — поворот вокруг латеральной оси, подбородок вперёд при +
      simulate     — false = только сегментация, без прогноза лица

    Возвращает session_id для отслеживания прогресса.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    scenario = validate_scenario(advance_mm, vertical_mm, lateral_mm, pitch_deg) if simulate else None

    # Create session
    session = create_session()
    session_dir = SESSION_DIR / session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    dicom_path = session_dir / "patient.dcm"
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 500MB)")
        
        with open(dicom_path, "wb") as f:
            f.write(content)
        
        session.dicom_path = str(dicom_path)
        session.status = "queued"
        session.scenario = scenario

    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        del sessions[session.session_id]
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Start processing in background
    asyncio.create_task(process_session(session))

    return {"session_id": session.session_id, "scenario": scenario}


# Physiological limits. A 40mm "advancement" is not a surgical plan, it is a typo,
# and it would silently produce a confident nonsense face.
SCENARIO_LIMITS_MM = 20.0
SCENARIO_LIMIT_DEG = 15.0


def validate_scenario(advance_mm: float, vertical_mm: float,
                      lateral_mm: float, pitch_deg: float) -> dict:
    for name, value, limit in (
        ("advance_mm", advance_mm, SCENARIO_LIMITS_MM),
        ("vertical_mm", vertical_mm, SCENARIO_LIMITS_MM),
        ("lateral_mm", lateral_mm, SCENARIO_LIMITS_MM),
        ("pitch_deg", pitch_deg, SCENARIO_LIMIT_DEG),
    ):
        if abs(value) > limit:
            raise HTTPException(
                status_code=422,
                detail=f"{name}={value} is outside the plausible range (±{limit})",
            )
    return {"advance_mm": advance_mm, "vertical_mm": vertical_mm,
            "lateral_mm": lateral_mm, "pitch_deg": pitch_deg}


async def process_session(session: Session):
    """Process session pipeline in background."""
    session_dir = SESSION_DIR / session.session_id
    
    def progress_callback(message: str, percentage: int):
        session.status = "processing"
        print(f"[{session.session_id}] {message} ({percentage}%)")
    
    try:
        session.status = "processing"
        
        result = await asyncio.to_thread(
            run_pipeline,
            session.session_id,
            session.dicom_path,
            session_dir,
            progress_callback,
            getattr(session, "scenario", None),
        )

        session.zip_path = result["zip_path"]
        session.simulation = result["simulation"]
        session.status = "completed"
        print(f"[{session.session_id}] Pipeline completed: {session.zip_path}")
        
    except PipelineError as e:
        session.status = "failed"
        session.error = f"{e.step}: {e.message}"
        print(f"[{session.session_id}] Pipeline failed at {e.step}: {e.message}")
        
    except Exception as e:
        session.status = "failed"
        session.error = str(e)
        print(f"[{session.session_id}] Pipeline failed: {str(e)}")


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get processing status for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    response = SessionStatus(
        session_id=session.session_id,
        status=session.status,
        message=None,
        error=session.error,
        simulation=session.simulation,
    )

    if session.status == "completed" and session.zip_path:
        response.download_url = f"/download/{session_id}"
    
    return response


@app.get("/download/{session_id}")
async def download_results(session_id: str):
    """Download results ZIP file."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "completed" or not session.zip_path:
        raise HTTPException(status_code=400, detail="Results not ready")
    
    zip_path = pathlib.Path(session.zip_path)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Mark as downloaded (but don't delete yet - keep for 1 week)
    session.status = "downloaded"
    
    return FileResponse(
        path=str(zip_path),
        filename=f"results_{session_id}.zip",
        media_type="application/zip",
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its files."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_dir = SESSION_DIR / session_id
    shutil.rmtree(session_dir, ignore_errors=True)
    del sessions[session_id]
    
    return {"success": True, "message": "Session deleted"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for demo management)."""
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "error": s.error,
            }
            for s in sessions.values()
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
