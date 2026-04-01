"""File management router."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from mas.config import get_config
from mas.utils.security import sanitize_filename

router = APIRouter(prefix="/api/files", tags=["files"])


def _upload_dir() -> Path:
    d = get_config().data_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("")
async def list_files():
    """List all uploaded files."""
    files = []
    for f in sorted(_upload_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "extension": f.suffix,
            })
    return files


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file."""
    safe_name = sanitize_filename(file.filename or "upload")
    path = _upload_dir() / safe_name
    content = await file.read()
    path.write_bytes(content)
    return {"name": safe_name, "size_bytes": len(content), "path": str(path)}


@router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete an uploaded file."""
    safe_name = sanitize_filename(filename)
    path = _upload_dir() / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": True}
