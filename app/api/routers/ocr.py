from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.python.convert_to_img import convert_pdf2img
from app.python.ocr import ocr_image, DEFAULT_PROMPT, DEFAULT_TIMEOUT, IMAGE_EXTS
from app.services.logger_service import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ocr"])

# Central directory for job state files (sibling of the app package)
_JOBS_DIR = Path(__file__).resolve().parents[3] / "jobs"

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"


# ---- Helpers ----

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read JSON: %s", path)
        return default


def get_project_root(project_id: str) -> Path:
    root = DATA_ROOT / project_id
    if not root.is_dir():
        logger.warning("Project not found: project_id=%s", project_id)
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return root


def append_audit(audit_path: Path, entry: dict) -> None:
    """Append one entry to audit.json. Never raises."""
    try:
        try:
            data = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            logger.exception("Failed to read audit.json: %s", audit_path)
            data = []
        data.append(entry)
        audit_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write audit.json: %s", audit_path)


# ---- Step functions ----

def step_convert_to_img(pdf_path: Path, out_dir: Path, file_id: str, zoom: float = 2.0) -> list[str]:
    ensure_dir(out_dir)
    return convert_pdf2img(
        input_file=str(pdf_path),
        out_dir=str(out_dir),
        base_override=file_id,
        zoom=zoom,
    )


def step_create_preview(
    pdf_path: Path,
    file_name: str,
    preview_dir: Path,
    file_id: str,
) -> tuple[list[str], str | None]:
    if not pdf_path.is_file() or not file_name.lower().endswith(".pdf"):
        return [], None
    try:
        preview_paths = step_convert_to_img(pdf_path, preview_dir, file_id)
        return preview_paths, None
    except Exception as e:
        logger.exception("Preview generation failed for file_id=%s: %s", file_id, e)
        return [], f"preview generation failed — {e}"


def step_ocr(
    pdf_path: Path,
    preview_paths: list[str],
    text_path: Path,
    ocr_model: str,
    ocr_prompt: str,
    ocr_timeout: float,
) -> list[str]:
    """
    Run OCR on preview images (PDF) or the file directly (image).
    Writes text to text_path incrementally (one page at a time).
    Returns list of per-page error messages.
    Skips if text_path already exists.
    """
    if text_path.is_file():
        return []

    images_to_ocr: list[str] = []
    if preview_paths:
        images_to_ocr = preview_paths
    elif pdf_path.is_file() and pdf_path.suffix.lower() in IMAGE_EXTS:
        images_to_ocr = [str(pdf_path)]

    if not images_to_ocr:
        return []

    ensure_dir(text_path.parent)
    ocr_errors: list[str] = []
    with text_path.open("w", encoding="utf-8") as fh:
        for i, img_path in enumerate(images_to_ocr, start=1):
            p = Path(img_path)
            if not p.is_file() or p.stat().st_size == 0:
                msg = f"image file missing or empty: {img_path}"
                ocr_errors.append(f"page {i}: {msg}")
                fh.write(f"===== PAGE {i} OCR FAILED =====\n{msg}\n\n")
                fh.flush()
                continue
            try:
                page_text = ocr_image(
                    model=ocr_model,
                    image_path=p,
                    prompt=ocr_prompt,
                    timeout=ocr_timeout,
                ).strip()
                fh.write(f"===== PAGE {i} =====\n{page_text}\n\n")
                fh.flush()
            except Exception as e:
                logger.exception("OCR failed on page %d (%s): %s", i, img_path, e)
                ocr_errors.append(f"page {i} ({img_path}): {e}")
                fh.write(f"===== PAGE {i} OCR FAILED =====\n{e}\n\n")
                fh.flush()
    return ocr_errors


# ---- Request model ----
class RunOCRRequest_Local(BaseModel):
    project_id: str
    file_ids: list[str]
    ocr_model: str = "ministral-3"
    ocr_prompt: str = DEFAULT_PROMPT
    ocr_timeout: float = DEFAULT_TIMEOUT
    force_rerun: bool = False


class RunOCRRequest(BaseModel):
    project_id: str
    file_ids: list[str]
    ocr_model: str = "ministral-3"
    ocr_prompt: str = DEFAULT_PROMPT
    ocr_timeout: float = DEFAULT_TIMEOUT
    force_rerun: bool = False
    callback: str | None = None  # Full URL (http/https) to POST results to when done



# ---- Endpoint ----

@router.post("/ocr_local")
def run_ocr_local(body: RunOCRRequest_Local):
    root = get_project_root(body.project_id)
    audit_path = root / "audit.json"

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    missing = [fid for fid in body.file_ids if fid not in file_index]
    if missing:
        logger.warning("OCR local — file IDs not found: project_id=%s missing=%s", body.project_id, missing)
        raise HTTPException(status_code=404, detail=f"File IDs not found in project: {missing}")

    results: list[dict] = []

    for fid in body.file_ids:
        file_record = file_index[fid]
        file_name = file_record.get("fileName", "")
        pdf_path = root / "files" / file_name
        text_path = root / "text" / f"{fid}.txt"

        # If force_rerun, delete existing text so OCR re-runs
        # if body.force_rerun and text_path.is_file():
        #     text_path.unlink()

        # Step 1: generate preview images
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_create_preview", "project_id": body.project_id, "file_id": fid})
        preview_paths, preview_err = step_create_preview(
            pdf_path=pdf_path,
            file_name=file_name,
            preview_dir=root / "previews" / fid,
            file_id=fid,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_create_preview", "project_id": body.project_id, "file_id": fid, "preview_count": len(preview_paths), "error": preview_err})

        # Step 2: OCR images → text file
        skipped = text_path.is_file()
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_ocr", "project_id": body.project_id, "file_id": fid, "skipped": skipped})
        ocr_errs = step_ocr(
            pdf_path=pdf_path,
            preview_paths=preview_paths,
            text_path=text_path,
            ocr_model=body.ocr_model,
            ocr_prompt=body.ocr_prompt,
            ocr_timeout=body.ocr_timeout,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_ocr", "project_id": body.project_id, "file_id": fid, "skipped": skipped, "status": "failed" if ocr_errs else "successful", "errors": ocr_errs})

        file_contents: list[dict] = []

        if text_path.is_file():
            file_contents.append({
                "contentType": "text/plain",
                "contentBase64": base64.b64encode(text_path.read_bytes()).decode("utf-8"),
            })

        for preview_path in preview_paths:
            p = Path(preview_path)
            if p.is_file():
                file_contents.append({
                    "contentType": "image/png",
                    "contentBase64": base64.b64encode(p.read_bytes()).decode("utf-8"),
                })

        results.append({
            "project_id": body.project_id,
            "file_id": fid,
            "ocr_skipped": skipped,
            "ocr_errors": ocr_errs,
            "preview_error": preview_err,
            "files": file_contents,
        })

    return JSONResponse(content=results)


# ---- Job state helpers ----

def _job_path(job_id: str) -> Path:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return _JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, state: dict) -> None:
    try:
        _job_path(job_id).write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        logger.exception("Failed to write job state: job_id=%s", job_id)


def _read_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---- Background worker ----

def _run_ocr_job(job_id: str, body: RunOCRRequest) -> None:
    root = DATA_ROOT / body.project_id
    audit_path = root / "audit.json"

    _write_job(job_id, {"job_id": job_id, "status": "running", "project_id": body.project_id})

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    results: list[dict] = []

    for fid in body.file_ids:
        file_record = file_index.get(fid)
        if not file_record:
            results.append({"project_id": body.project_id, "file_id": fid, "error": "file not found"})
            continue

        file_name = file_record.get("fileName", "")
        pdf_path = root / "files" / file_name
        text_path = root / "text" / f"{fid}.txt"

        # if body.force_rerun and text_path.is_file():
        #     text_path.unlink()

        append_audit(audit_path, {"ts": now_iso(), "action": "start step_create_preview", "job_id": job_id, "project_id": body.project_id, "file_id": fid})
        preview_paths, preview_err = step_create_preview(
            pdf_path=pdf_path,
            file_name=file_name,
            preview_dir=root / "previews" / fid,
            file_id=fid,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_create_preview", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "preview_count": len(preview_paths), "error": preview_err})

        skipped = text_path.is_file()
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_ocr", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "skipped": skipped})
        ocr_errs = step_ocr(
            pdf_path=pdf_path,
            preview_paths=preview_paths,
            text_path=text_path,
            ocr_model=body.ocr_model,
            ocr_prompt=body.ocr_prompt,
            ocr_timeout=body.ocr_timeout,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_ocr", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "skipped": skipped, "status": "failed" if ocr_errs else "successful", "errors": ocr_errs})

        file_contents: list[dict] = []
        if text_path.is_file():
            file_contents.append({
                "contentType": "text/plain",
                "contentBase64": base64.b64encode(text_path.read_bytes()).decode("utf-8"),
            })
        for preview_path in preview_paths:
            p = Path(preview_path)
            if p.is_file():
                file_contents.append({
                    "contentType": "image/png",
                    "contentBase64": base64.b64encode(p.read_bytes()).decode("utf-8"),
                })

        results.append({
            "project_id": body.project_id,
            "file_id": fid,
            "ocr_skipped": skipped,
            "ocr_errors": ocr_errs,
            "preview_error": preview_err,
            "files": file_contents,
        })

    # Persist completed results so polling endpoint can serve them
    _write_job(job_id, {"job_id": job_id, "status": "completed", "project_id": body.project_id, "results": results})

    if body.callback:
        payload = {"job_id": job_id, "status": "completed", "results": results}
        try:
            with httpx.Client(timeout=30) as client:
                client.post(body.callback, json=payload)
        except Exception as e:
            logger.exception("Callback POST failed for job_id=%s url=%s: %s", job_id, body.callback, e)
            append_audit(audit_path, {"ts": now_iso(), "action": "callback_failed", "job_id": job_id, "project_id": body.project_id, "error": str(e)})


# ---- Endpoints ----

@router.post("/ocr")
def run_ocr(body: RunOCRRequest, background_tasks: BackgroundTasks):
    """Queue an OCR job and return immediately with a job_id.
    Poll GET /api/ocr/jobs/{job_id} to retrieve results when done."""
    root = get_project_root(body.project_id)

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    missing = [fid for fid in body.file_ids if fid not in file_index]
    if missing:
        logger.warning("OCR — file IDs not found: project_id=%s missing=%s", body.project_id, missing)
        raise HTTPException(status_code=404, detail=f"File IDs not found in project: {missing}")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_ocr_job, job_id, body)
    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@router.get("/ocr/jobs/{job_id}")
def get_ocr_job(job_id: str):
    """Poll for OCR job status and results.
    Returns {"status": "queued"|"running"|"completed"} with results once done."""
    state = _read_job(job_id)
    if state is None:
        logger.warning("Job not found: job_id=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JSONResponse(content=state)


class UpdateTextRequest(BaseModel):
    project_id: str
    file_id: str
    text: str

@router.post("/update_text")
def update_text(body: UpdateTextRequest):
    root = get_project_root(body.project_id)
    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", [])}
    if body.file_id not in file_index:
        logger.warning("update_text — file ID not found: project_id=%s file_id=%s", body.project_id, body.file_id)
        raise HTTPException(status_code=404, detail=f"File ID not found in project: {body.file_id}")    
    
    text_path = root / "text" / f"{body.file_id}.txt"
    ensure_dir(text_path.parent)
    text_path.write_text(body.text, encoding="utf-8")
    return JSONResponse(content={"project_id": body.project_id, "file_id": body.file_id, "updated": True,  "status": "success"})