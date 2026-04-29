from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.python.llm_extract_core import answer_questions_json, answer_questions_json_chunked
from app.services.logger_service import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"

TOKEN_THRESHOLD = 15_000  # Lowered: chunk documents over ~15k tokens to avoid Ollama timeouts

MODEL: str = "ministral-3"  # Ollama model name for extraction; use a smaller/faster model if you have many files or long documents
CONTEXT_NOTE: str = ""
FORCE_CHUNKING: bool = True
CHUNK_CHARS: int = 12_000
OVERLAP_CHARS: int = 800


# ---- Helpers ----

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_display() -> str:
    return datetime.now(timezone.utc).strftime("%-Y-%m-%d %-I:%M %p")


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(5)}"


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


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def normalize_answers(result: Any, questions: list[str]) -> dict[str, Any]:
    """Normalize extraction result to {question: {value, confidence}} for all questions."""
    null_entry: dict[str, Any] = {"value": None, "confidence": 0.0}
    if not isinstance(result, dict):
        return {q: null_entry for q in questions}
    out: dict[str, Any] = {}
    for q in questions:
        entry = result.get(q, null_entry)
        if isinstance(entry, dict) and "value" in entry:
            out[q] = {
                "value": entry.get("value"),
                "confidence": float(entry.get("confidence", 0.0)),
            }
        else:
            out[q] = null_entry
    return out


def build_context_note(base: str, standard: dict | None, structure_text: str | None) -> str:
    parts = []
    if base.strip():
        parts.append(base.strip())
    if structure_text and structure_text.strip():
        parts.append("File-type structure / layout notes:\n" + structure_text.strip())
    if standard and isinstance(standard.get("fields"), list):
        lines = ["Data standard (fields + definitions):"]
        for f in standard["fields"]:
            fn = str(f.get("user_question") or f.get("field") or "").strip()
            if not fn:
                continue
            desc = str(f.get("description") or f.get("definition") or "").strip()
            dtype = str(f.get("dataType") or "").strip()
            line = f"- {fn}: {desc}"
            if dtype:
                line += f" [{dtype}]"
            lines.append(line)
        parts.append("\n".join(lines))
    return "\n\n".join(parts).strip()


# ---- Step function ----

def step_extraction(
    doc_text: str,
    questions: list[str],
    model: str,
    ctx_full: str,
    force_chunking: bool,
    token_threshold: int,
    chunk_chars: int,
    overlap_chars: int,
    timeout: float = 600.0,
) -> tuple[dict[str, Any], bool]:
    """
    Run LLM extraction on document text.
    Returns (normalized_answers, use_chunked).
    Raises on failure so the caller can record the error.
    """
    tok_est = estimate_tokens(doc_text)
    use_chunked = force_chunking or (tok_est > token_threshold)

    if use_chunked:
        raw = answer_questions_json_chunked(
            model=model,
            document_text=doc_text,
            questions=questions,
            context_note=ctx_full,
            max_chars=chunk_chars,
            overlap=overlap_chars,
            timeout=timeout,
        )
    else:
        raw = answer_questions_json(
            model=model,
            document_text=doc_text,
            questions=questions,
            context_note=ctx_full,
            timeout=timeout,
        )

    return normalize_answers(raw, questions), use_chunked


# ---- Request model ----

class StandardField(BaseModel):
    field: str = ""
    user_question: str = ""
    definition: str = ""
    description: str = ""
    dataType: str = ""
    required: bool = False
    format: str = ""
    allowed_values: str = ""


class StandardSchema(BaseModel):
    fields: list[StandardField] = []
    version: int | None = None


class RunExtractRequest(BaseModel):
    project_id: str
    file_ids: list[str]
    standard: StandardSchema | None = None  # Inline standard JSON (same schema as standard_v1.json)
    structure: str | None = None  # Direct text content of the structure file
    extraction_timeout: float = 600.0  # Seconds; increase for slow/CPU-only Ollama instances
    callback_url: str | None = None  # If set, run async and POST result to this URL when done


# ---- Core extraction logic ----

def _do_extraction(body: RunExtractRequest, run_id: str | None = None) -> dict:
    """Run the full extraction synchronously and return the flat response dict."""
    root = get_project_root(body.project_id)
    audit_path = root / "audit.json"

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    missing = [fid for fid in body.file_ids if fid not in file_index]
    if missing:
        logger.warning("run_extract — file IDs not found: project_id=%s missing=%s", body.project_id, missing)
        raise HTTPException(status_code=404, detail=f"File IDs not found in project: {missing}")

    standard: dict | None = body.standard.model_dump() if body.standard else None
    structure_text: str | None = body.structure

    questions: list[str] = []
    if standard and isinstance(standard.get("fields"), list):
        questions = [
            str(f.get("user_question") or f.get("field") or "").strip()
            for f in standard["fields"]
            if str(f.get("user_question") or f.get("field") or "").strip()
        ]
    if not questions:
        raw_fields = standard.get("fields") if standard else None
        logger.warning(
            "run_extract — no questions: project_id=%s standard_none=%s field_count=%s raw_fields=%s",
            body.project_id,
            standard is None,
            len(raw_fields) if isinstance(raw_fields, list) else "N/A",
            raw_fields,
        )
        raise HTTPException(
            status_code=400,
            detail=f"No questions to extract. standard_none={standard is None}, field_count={len(raw_fields) if isinstance(raw_fields, list) else 'N/A'}, raw_fields={raw_fields}",
        )

    ctx_full = build_context_note(CONTEXT_NOTE, standard, structure_text)

    run_id = run_id or new_run_id()
    run_record: dict = {
        "run_id": run_id,
        "created_at": now_iso(),
        "project_id": body.project_id,
        "model": MODEL,
        "context_note": ctx_full,
        "standard": body.standard.model_dump() if body.standard else None,
        "structure": body.structure,
        "files": [],
        "outputs": {},
        "params": {
            "force_chunking": FORCE_CHUNKING,
            "token_threshold": TOKEN_THRESHOLD,
            "chunk_chars": CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
        },
    }

    errors: list[str] = []

    for fid in body.file_ids:
        file_record = file_index[fid]
        file_name = file_record.get("fileName", "")
        text_path = root / "text" / f"{fid}.txt"
        preview_paths: list[str] = []

        preview_dir = root / "previews" / fid
        if preview_dir.is_dir():
            preview_paths = sorted(str(p) for p in preview_dir.iterdir() if p.suffix.lower() == ".png")

        if not text_path.is_file():
            run_record["outputs"][fid] = {q: {"value": None, "confidence": 0.0} for q in questions}
            errors.append(f"{fid}: no text layer found — run OCR first via POST /api/ocr")
            run_record["files"].append({
                "file_id": fid,
                "fileName": file_name,
                "token_estimate": None,
                "chunked": None,
                "preview_count": len(preview_paths),
                "preview_paths": preview_paths,
            })
            continue

        doc_text = text_path.read_text(encoding="utf-8", errors="replace")
        tok_est = estimate_tokens(doc_text)
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_extraction", "project_id": body.project_id, "file_id": fid})
        try:
            answers, use_chunked = step_extraction(
                doc_text=doc_text,
                questions=questions,
                model=MODEL,
                ctx_full=ctx_full,
                force_chunking=FORCE_CHUNKING,
                token_threshold=TOKEN_THRESHOLD,
                chunk_chars=CHUNK_CHARS,
                overlap_chars=OVERLAP_CHARS,
                timeout=body.extraction_timeout,
            )
        except Exception as e:
            logger.exception("Extraction failed for project_id=%s file_id=%s: %s", body.project_id, fid, e)
            append_audit(audit_path, {"ts": now_iso(), "action": "complete step_extraction", "project_id": body.project_id, "file_id": fid, "status": "failed", "error": str(e)})
            run_record["outputs"][fid] = {q: {"value": None, "confidence": 0.0} for q in questions}
            errors.append(f"{fid}: extraction failed — {e}")
            run_record["files"].append({
                "file_id": fid,
                "fileName": file_name,
                "token_estimate": tok_est,
                "chunked": None,
                "preview_count": len(preview_paths),
                "preview_paths": preview_paths,
            })
            continue
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_extraction", "project_id": body.project_id, "file_id": fid, "status": "successful"})

        run_record["outputs"][fid] = answers
        run_record["files"].append({
            "file_id": fid,
            "fileName": file_name,
            "token_estimate": tok_est,
            "chunked": use_chunked,
            "preview_count": len(preview_paths),
            "preview_paths": preview_paths,
        })

    if errors:
        run_record["errors"] = errors

    runs_dir = root / "runs"
    ensure_dir(runs_dir)
    write_json(runs_dir / f"run_{run_id}.json", run_record)

    manifest["lastModified"] = now_display()
    write_json(root / "project.json", manifest)

    append_audit(audit_path, {
        "ts": now_iso(),
        "action": "run.save",
        "project_id": body.project_id,
        "run_id": run_id,
        "file_ids": body.file_ids,
        "status": "successful",
    })

    flat_outputs = [
        {"question": q, "answer": v["value"], "confidence": v["confidence"]}
        for answers in run_record["outputs"].values()
        for q, v in answers.items()
    ]

    return {
        "run_id": run_id,
        "project_id": body.project_id,
        "model": MODEL,
        "file_ids": body.file_ids,
        "outputs": flat_outputs,
        "errors": errors,
    }


def _run_extraction_in_thread(body: RunExtractRequest, run_id: str, callback_url: str | None) -> None:
    """Blocking work: runs on a worker thread so the event loop stays free."""
    root = DATA_ROOT / body.project_id
    runs_dir = root / "runs"
    pending_path = runs_dir / f"run_{run_id}.pending"

    result: dict
    try:
        result = _do_extraction(body, run_id=run_id)
    except Exception as e:
        logger.exception("Background extraction failed for project_id=%s: %s", body.project_id, e)
        result = {
            "run_id": run_id,
            "project_id": body.project_id,
            "file_ids": body.file_ids,
            "outputs": [],
            "errors": [str(e)],
        }
        try:
            ensure_dir(runs_dir)
            write_json(runs_dir / f"run_{run_id}.json", result)
        except Exception:
            logger.exception("Failed to write failed run file for run_id=%s", run_id)
    finally:
        try:
            pending_path.unlink(missing_ok=True)
        except Exception:
            pass

    if callback_url:
        try:
            with httpx.Client(timeout=30.0, verify=False) as client:
                client.post(callback_url, json=result)
            logger.info("Callback posted for project_id=%s to %s", body.project_id, callback_url)
        except Exception:
            logger.exception("Failed to POST callback to %s for project_id=%s", callback_url, body.project_id)


async def _background_extract_and_callback(body: RunExtractRequest, run_id: str, callback_url: str | None) -> None:
    """Async background task: offloads blocking work to a thread."""
    await anyio.to_thread.run_sync(
        lambda: _run_extraction_in_thread(body, run_id, callback_url),
        cancellable=False,
    )


# ---- Endpoints ----

@router.post("")
async def run_extract(body: RunExtractRequest, background_tasks: BackgroundTasks):
    run_id = new_run_id()

    root = DATA_ROOT / body.project_id
    runs_dir = root / "runs"
    try:
        await anyio.to_thread.run_sync(lambda: (ensure_dir(runs_dir), (runs_dir / f"run_{run_id}.pending").write_text("pending", encoding="utf-8")))
    except Exception:
        logger.exception("Failed to write pending marker for run_id=%s", run_id)

    field_count = len(body.standard.fields) if body.standard else 0
    logger.info(
        "Async extraction accepted for project_id=%s file_ids=%s run_id=%s standard_fields=%d",
        body.project_id, body.file_ids, run_id, field_count,
    )
    if field_count == 0:
        logger.warning("run_extract — standard has no fields: project_id=%s standard=%s", body.project_id, body.standard)
    background_tasks.add_task(_background_extract_and_callback, body, run_id, body.callback_url)
    return JSONResponse(content={"status": "accepted", "run_id": run_id, "project_id": body.project_id, "file_ids": body.file_ids})


@router.get("/{project_id}/{run_id}")
async def get_run_status(project_id: str, run_id: str):
    """Poll endpoint: returns pending/complete/failed status for an async run."""
    root = DATA_ROOT / project_id
    pending_path = root / "runs" / f"run_{run_id}.pending"
    run_path = root / "runs" / f"run_{run_id}.json"

    if pending_path.exists():
        return JSONResponse(content={"status": "pending", "run_id": run_id})
    if run_path.exists():
        data = read_json(run_path, {})
        errors = data.get("errors", [])
        raw_outputs = data.get("outputs", {})

        # run_record saves outputs as {file_id: {question: {value, confidence}}}
        # flatten to the array format Node.js expects: [{question, answer, confidence}]
        if isinstance(raw_outputs, dict):
            flat_outputs = [
                {"question": q, "answer": v.get("value"), "confidence": v.get("confidence", 0.0)}
                for answers in raw_outputs.values()
                if isinstance(answers, dict)
                for q, v in answers.items()
                if isinstance(v, dict)
            ]
        else:
            flat_outputs = raw_outputs  # already flat (error-case fallback)

        if errors and not flat_outputs:
            return JSONResponse(content={"status": "failed", "run_id": run_id, "outputs": flat_outputs, "errors": errors})
        return JSONResponse(content={"status": "complete", "run_id": run_id, "outputs": flat_outputs, "errors": errors})
    return JSONResponse(status_code=404, content={"status": "not_found", "run_id": run_id})