"""File management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/files", tags=["files"])


def _upload_dir(request: Request) -> Path:
    d = request.app.state.platform.config.data_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("")
async def list_files(request: Request):
    upload_dir = _upload_dir(request)
    return [
        {
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "extension": f.suffix,
        }
        for f in sorted(upload_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if f.is_file()
    ]


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Sanitize filename
    from pathlib import PurePosixPath, PureWindowsPath
    name = PurePosixPath(file.filename or "upload").name
    name = PureWindowsPath(name).name
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = _upload_dir(request) / name
    content = await file.read()
    path.write_bytes(content)
    return {"name": name, "size_bytes": len(content)}


@router.delete("/{filename}")
async def delete_file(request: Request, filename: str):
    path = _upload_dir(request) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": True}
