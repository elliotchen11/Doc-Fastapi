# llm_extract_core.py (patch)

from __future__ import annotations
import json
import re
from typing import List, Dict, Optional, Any
from ollama import Client
import os

try:
    from app.services.logger_service import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

DEFAULT_TIMEOUT = 120.0


SYSTEM_PROMPT = """You are a data extraction assistant. Extract specific field values from documents.

Rules:
- Extract the EXACT value as it appears in the document — do not paraphrase, summarize, or interpret.
- If a field value is not present in the document, return null.
- Keep values short and precise: names, dates, numbers, short phrases — not paragraphs.
- Do NOT describe the document. Do NOT write summaries or explanations.
- Fill in the JSON template provided by the user. Do not add or remove keys.
- CRITICAL: Every answer MUST follow exactly this shape: {"value": "<string or null>", "confidence": <0.0-1.0>}
- CRITICAL: Never nest objects inside a value. The value field must always be a plain string or null.
"""

def extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of markdown/codefences/prose."""
    if not text:
        return None

    # 1) Code fence: ```json { ... } ```
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) First balanced {...} block, respecting strings/escapes
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return None


def safe_load_json_from_model(text: str) -> Dict[str, Any]:
    snippet = extract_json_object(text)
    if not snippet:
        # Model may have returned key-value pairs without outer {} — try wrapping only if it
        # looks like JSON (starts with a quoted key), not prose or markdown.
        stripped = text.strip().rstrip(",")
        if stripped.startswith('"'):
            snippet = "{" + stripped + "}"
        if not snippet:
            logger.warning("safe_load_json: no JSON object found in model output (len=%d)", len(text))
            return {}
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        logger.warning("safe_load_json: JSON parse failed — %s | snippet[:200]=%s", e, snippet[:200])
        return {}


def _repair_to_json(model: str, bad_text: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    client = Client(timeout=timeout)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "user", "content": f"Convert this to valid JSON only:\n\n{bad_text}"},
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 4096},
    )
    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    return parsed if isinstance(parsed, dict) else {}


def _key_norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.lower().strip())


def _find_in_json(obj: Any, question: str) -> Any:
    """Recursively find a scalar value whose key fuzzy-matches the question."""
    q = _key_norm(question)
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = _key_norm(k)
            if kn == q or kn.endswith("_" + q) or q in kn or kn in q:
                if isinstance(v, dict) and "value" in v:
                    return v
                if not isinstance(v, (dict, list)):
                    return v
        for v in obj.values():
            result = _find_in_json(v, question)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_in_json(item, question)
            if result is not None:
                return result
    return None


def _normalize_question_for_model(q: str) -> str:
    # keeps user-visible key unchanged elsewhere; this is only for the model prompt
    q2 = (q or "").strip()
    q2 = re.sub(r"\s+", " ", q2)
    q2 = q2.rstrip(":").strip()
    return q2


def _extract_field(raw: Any) -> Dict[str, Any]:
    """Normalize a single answer entry to {value, confidence}."""
    if isinstance(raw, dict):
        if "value" in raw:
            value = raw.get("value")
            if "confidence" in raw:
                confidence = raw.get("confidence")
            else:
                # Model didn't supply confidence — derive from whether a value was found
                confidence = 1.0 if value is not None else 0.0
        else:
            # LLM returned a nested object instead of {value, confidence} —
            # serialize it so data is preserved rather than dropped.
            value = json.dumps(raw, ensure_ascii=False)
            confidence = 0.5
    else:
        # Old-style plain string answer — treat as full confidence if non-null
        value = raw
        confidence = 0.0 if raw is None else 1.0

    if isinstance(value, str):
        value = value.strip() or None
    elif value is not None:
        value = str(value).strip() or None

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {"value": value, "confidence": confidence}


def answer_questions_json(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Any]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]

    if not orig_questions:
        return {}

    client = Client(timeout=timeout)
    out: Dict[str, Dict[str, Any]] = {}

    # Ask one question at a time with a minimal {value} schema.
    # Batch templates are ignored by small models — per-question requests are reliable.
    for q in orig_questions:
        user_prompt = (
            f'Extract the value for: "{q}"\n\n'
            f"Context: {context_note}\n\n"
            "Document:\n<doc>\n"
            f"{document_text}\n"
            "</doc>\n\n"
            f'Return ONLY this JSON: {{"value": "the exact value"}} or {{"value": null}} if not found.'
        )
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": temperature, "num_predict": 2048},
            )
            content = (getattr(getattr(resp, "message", None), "content", None) or "").strip()
            logger.debug("LLM answer for %r (first 300 chars): %s", q, content[:300])

            parsed = safe_load_json_from_model(content)
            if isinstance(parsed, dict) and "value" in parsed:
                out[q] = _extract_field(parsed)
            elif isinstance(parsed, dict) and parsed:
                # Try to find the field value by key matching first
                matched = _find_in_json(parsed, q)
                if matched is not None:
                    out[q] = _extract_field(matched)
                elif len(parsed) == 1:
                    # Single-key object like {"summary": "..."} — use the value directly
                    sole_value = next(iter(parsed.values()))
                    if isinstance(sole_value, str):
                        v = sole_value.strip() or None
                        out[q] = {"value": v, "confidence": 0.5 if v else 0.0}
                    else:
                        out[q] = _extract_field(sole_value)
                else:
                    out[q] = _extract_field(json.dumps(parsed))
            else:
                out[q] = {"value": None, "confidence": 0.0}
        except Exception as e:
            logger.warning("LLM call failed for question %r: %s", q, e)
            out[q] = {"value": None, "confidence": 0.0}

    return out


def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> list[str]:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    i = 0
    while i < len(s):
        j = min(len(s), i + max_chars)
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(0, j - overlap)
    return chunks


def answer_questions_json_chunked(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Any]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Dict[str, Any]] = {q: {"value": None, "confidence": 0.0} for q in qs}
    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        partial = answer_questions_json(
            model=model,
            document_text=f"[CHUNK {idx}/{len(chunks)}]\n{ch}",
            questions=qs,
            context_note=context_note + " Answer only if the information is present in this chunk.",
            temperature=0.1,
            timeout=timeout,
        )
        for q in qs:
            new_entry = partial.get(q, {"value": None, "confidence": 0.0})
            if new_entry.get("confidence", 0.0) > merged[q].get("confidence", 0.0):
                merged[q] = new_entry

    return merged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test answer_questions_json from the CLI")
    parser.add_argument("--model", default="ministral-3")
    parser.add_argument("--text-file", required=True, help="Path to the OCR text file")
    parser.add_argument("--questions", required=True, nargs="+", help="One or more field names to extract")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--chunked", action="store_true", help="Force chunked mode")
    parser.add_argument("--raw", action="store_true", help="Print raw model output before parsing")
    args = parser.parse_args()
    doc_text = open(args.text_file, encoding="utf-8").read()
    print(f"[info] model={args.model}  timeout={args.timeout}s  chars={len(doc_text)}  chunked={args.chunked}")
    print(f"[info] questions={args.questions}\n")
    if args.raw:
        orig_questions = [q.strip() for q in args.questions if q and q.strip()]
        q_items = [{"id": f"q{i}", "question": q} for i, q in enumerate(orig_questions, 1)]
        user_prompt = (
            "Context note:\nThe information might be spread out through the entire document.\n\n"
            f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
            "Document text:\n<doc>\n"
            f"{doc_text}\n"
            "</doc>\n\nReturn JSON now."
        )
        client = Client(timeout=args.timeout)
        resp = client.chat(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 4096},
        )
        content = getattr(getattr(resp, "message", None), "content", None) or ""
        print("[raw model output]")
        print(content)
        print()
        print("[parsed]")
        print(json.dumps(safe_load_json_from_model(content), indent=2, ensure_ascii=False))
    elif args.chunked:
        result = answer_questions_json_chunked(
            model=args.model,
            document_text=doc_text,
            questions=args.questions,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = answer_questions_json(
            model=args.model,
            document_text=doc_text,
            questions=args.questions,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
