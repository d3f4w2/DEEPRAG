from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
import json
import os
import mimetypes
from pathlib import Path
import signal
import re
from typing import Any, AsyncIterator, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv

from backend.config import settings
from backend.models import (
    ChatRequest, FileRetrievalRequest, FileRetrievalResponse,
    KnowledgeBaseInfo, HealthResponse, EvalStartRequest,
    VoiceDraftRequest, VoiceDraftResponse, VoiceIngestRequest, VoiceIngestResponse,
    ImageDraftResponse, ImageIngestResponse
)
from backend.knowledge_base import knowledge_base
from backend.llm_provider import LLMProvider
from backend.evaluation_manager import evaluation_manager
from backend.image_processing import (
    inspect_image_bytes,
    parse_tags_text,
    run_paddle_ocr,
    run_vision_analysis,
)
from backend.prompts import (
    create_react_system_prompt,
    create_retrieve_sections_tool,
    create_search_paths_tool,
    create_system_prompt,
    process_tool_calls,
)
from backend.react_handler import handle_react_mode

app = FastAPI(
    title="Deep RAG",
    version="1.0.0",
    description="A Deep RAG system that teaches AI to truly understand your knowledge base"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_ROUTE_PLAN = {
    "top_k": 6,
    "max_sections_per_file": 2,
    "min_file_paths": 4,
}
AUTO_RETRIEVAL_TOP_K_BOOST = 2
MAX_FINAL_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_SNIPPET_LENGTH = 280
VOICE_NOTES_DIR = "Voice-Notes"
IMAGE_NOTES_DIR = "Image-Notes"
IMAGE_ASSETS_DIR = "assets"
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FALLBACK_NO_EVIDENCE_ANSWER = (
    "我无法给出有证据支撑的回答。已自动再检索一轮，但仍未找到可引用片段。"
)
EVIDENCE_SECTION_PATTERN = re.compile(r"(?is)\n#{2,3}\s*(证据|evidence)\s*\n.*$")
EVIDENCE_BLOCK_PATTERN = re.compile(
    r"\[\[FILE:(?P<path>.+?)\s*\|\s*SECTION:(?P<section>\d+)\s*\|\s*SCORE:(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*HEADING:(?P<heading>.*?)\]\]\s*(?P<body>.*?)(?=(?:\n\n==========\n\n|\n\n----------\n\n|\Z))",
    re.DOTALL,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_candidate_paths(search_output: str) -> List[str]:
    parsed = _extract_json_object(search_output or "")
    candidates = parsed.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    paths: List[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if path:
            paths.append(path)
    return paths


def _extract_evidence_entries(section_output: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for match in EVIDENCE_BLOCK_PATTERN.finditer(section_output or ""):
        file_path = _normalize_whitespace(match.group("path"))
        snippet = _normalize_whitespace(match.group("body"))
        if not file_path or not snippet:
            continue
        if len(snippet) > MAX_EVIDENCE_SNIPPET_LENGTH:
            snippet = snippet[:MAX_EVIDENCE_SNIPPET_LENGTH].rstrip() + "..."
        snippet = snippet.replace('"', "'")
        entries.append({"file_path": file_path, "snippet": snippet})
    return entries


def _merge_evidence_pool(
    current_pool: List[Dict[str, str]], new_entries: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    merged = list(current_pool)
    seen = {(item.get("file_path", ""), item.get("snippet", "")) for item in merged}
    for entry in new_entries:
        key = (entry.get("file_path", ""), entry.get("snippet", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    return merged


def _strip_existing_evidence_section(answer: str) -> str:
    if not answer:
        return ""
    text = answer.strip()
    text = EVIDENCE_SECTION_PATTERN.sub("", text)
    return text.strip()


def _format_answer_with_evidence(answer: str, evidence_pool: List[Dict[str, str]]) -> str:
    if not evidence_pool:
        return FALLBACK_NO_EVIDENCE_ANSWER

    clean_answer = _strip_existing_evidence_section(answer)
    if not clean_answer:
        clean_answer = "基于检索到的知识库证据，结论如下。"

    lines = [clean_answer, "", "### 证据"]
    for idx, item in enumerate(evidence_pool[:MAX_FINAL_EVIDENCE_ITEMS], start=1):
        file_path = item.get("file_path", "").strip() or "unknown"
        snippet = item.get("snippet", "").strip() or "(空片段)"
        lines.append(f"{idx}. 来源文件: `{file_path}`")
        lines.append(f"   证据片段: \"{snippet}\"")

    return "\n".join(lines).strip()


def _build_tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = default
    return max(low, min(iv, high))


def _safe_parse_tool_args(arguments: str) -> Dict[str, Any]:
    return _extract_json_object(arguments or "")


async def _llm_expand_retrieval_query(provider: LLMProvider, user_query: str) -> Dict[str, Any]:
    fallback = {
        "expanded_query": (user_query or "").strip(),
        "image_intent": False,
        "hints": [],
    }
    query = (user_query or "").strip()
    if not query:
        return fallback

    messages = [
        {
            "role": "system",
            "content": (
                "You optimize retrieval queries for bilingual (Chinese/English) lexical search. "
                "Return strict JSON only: "
                "{\"expanded_query\": string, \"image_intent\": boolean, \"hints\": string[]}. "
                "Rules: "
                "1) Preserve original intent. "
                "2) Add concise Chinese+English synonyms and key entities for retrieval. "
                "3) If query is about image/photo/portrait/OCR/scene, set image_intent=true. "
                "4) expanded_query should be one line."
            ),
        },
        {"role": "user", "content": query},
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(messages=messages, tools=None, stream=False):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception:
        return fallback

    parsed = _extract_json_object(raw_text)
    expanded_query = _normalize_whitespace(str(parsed.get("expanded_query") or query))
    image_intent = bool(parsed.get("image_intent"))
    hints: List[str] = []
    raw_hints = parsed.get("hints")
    if isinstance(raw_hints, list):
        for item in raw_hints:
            hint = _normalize_whitespace(str(item))
            if hint and hint not in hints:
                hints.append(hint)

    return {
        "expanded_query": expanded_query or query,
        "image_intent": image_intent,
        "hints": hints[:16],
    }


def _slugify_filename(value: str, fallback: str = "speaker") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (value or "").strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _to_local_iso(raw_value: Optional[str]) -> str:
    if not raw_value:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    raw = raw_value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().isoformat(timespec="seconds")


def _guess_image_suffix(filename: str, content_type: str, image_format: str) -> str:
    for candidate in [
        Path(filename or "").suffix.lower(),
        mimetypes.guess_extension(content_type or "") or "",
        f".{(image_format or '').lower()}",
    ]:
        if candidate in ALLOWED_IMAGE_SUFFIXES:
            return candidate
    return ".png"


async def _read_upload_image_bytes(file: UploadFile) -> bytes:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")
    if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds size limit ({MAX_IMAGE_UPLOAD_BYTES} bytes).",
        )
    return image_bytes


def _fallback_voice_summary(transcript: str) -> str:
    normalized = _normalize_whitespace(transcript)
    if not normalized:
        return ""
    if len(normalized) <= 60:
        return normalized
    return normalized[:60].rstrip() + "..."


def _parse_voice_draft_response(raw_text: str) -> tuple[str, str, str]:
    """Parse draft response from multiple possible formats.

    Returns: (polished_text, summary, parse_note)
    """
    raw = (raw_text or "").strip()
    if not raw:
        return "", "", "empty response"

    # 1) Direct JSON object.
    parsed = _extract_json_object(raw)
    polished = _normalize_whitespace(str(parsed.get("polished_text") or ""))
    summary = _normalize_whitespace(str(parsed.get("summary") or ""))
    if polished or summary:
        return polished, summary, "json"

    # 2) JSON inside fenced code block.
    fence = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence:
        parsed = _extract_json_object(fence.group(1))
        polished = _normalize_whitespace(str(parsed.get("polished_text") or ""))
        summary = _normalize_whitespace(str(parsed.get("summary") or ""))
        if polished or summary:
            return polished, summary, "json-fence"

    # 3) Tagged format.
    tag_pol = re.search(r"<polished_text>\s*([\s\S]*?)\s*</polished_text>", raw, flags=re.IGNORECASE)
    tag_sum = re.search(r"<summary>\s*([\s\S]*?)\s*</summary>", raw, flags=re.IGNORECASE)
    polished = _normalize_whitespace(tag_pol.group(1) if tag_pol else "")
    summary = _normalize_whitespace(tag_sum.group(1) if tag_sum else "")
    if polished or summary:
        return polished, summary, "xml-tags"

    # 4) Label format: English.
    m_en = re.search(
        r"POLISHED_TEXT\s*[:：]\s*([\s\S]*?)(?:\n+\s*SUMMARY\s*[:：]\s*([\s\S]*))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_en:
        polished = _normalize_whitespace(m_en.group(1) or "")
        summary = _normalize_whitespace(m_en.group(2) or "")
        if polished or summary:
            return polished, summary, "labels-en"

    # 5) Label format: Chinese.
    m_zh = re.search(
        r"(?:润色文本|纠正文本|草稿)\s*[:：]\s*([\s\S]*?)(?:\n+\s*(?:摘要|总结)\s*[:：]\s*([\s\S]*))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_zh:
        polished = _normalize_whitespace(m_zh.group(1) or "")
        summary = _normalize_whitespace(m_zh.group(2) or "")
        if polished or summary:
            return polished, summary, "labels-zh"

    # 6) Free-form fallback: treat entire output as polished text.
    stripped = re.sub(r"```[\s\S]*?```", "", raw).strip()
    polished = _normalize_whitespace(stripped)
    if polished:
        return polished, "", "free-form"

    return "", "", "unparsed"


async def _llm_voice_summary(
    provider: LLMProvider,
    transcript: str,
    polished_text: str,
    author: str,
) -> str:
    summary_messages = [
        {
            "role": "system",
            "content": (
                "Summarize the core meaning in ONE concise sentence. "
                "Output plain text only. No markdown, no bullet points, no labels."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "author": author,
                    "transcript": transcript,
                    "polished_text": polished_text,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=summary_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw += str(chunk.get("content") or "")
    except Exception:
        return ""

    plain = _normalize_whitespace(raw)
    plain = re.sub(r"^(?:summary|摘要|总结)\s*[:：]\s*", "", plain, flags=re.IGNORECASE)
    return plain


async def _llm_voice_draft(
    provider: LLMProvider,
    transcript: str,
    author: str,
) -> Dict[str, str]:
    fallback_polished = _normalize_whitespace(transcript)
    fallback_summary = _fallback_voice_summary(fallback_polished)

    draft_messages = [
        {
            "role": "system",
            "content": (
                "You are a voice note editor. Rewrite colloquial transcript into clean readable text "
                "without changing facts, then provide one-sentence summary. "
                "Preferred output format (plain text, no markdown):\n"
                "POLISHED_TEXT: <rewritten text>\n"
                "SUMMARY: <one sentence>\n"
                "If you return JSON, schema must be {\"polished_text\": string, \"summary\": string}. "
                "Keep original facts, remove obvious fillers and disfluencies, and keep the same language "
                "as transcript. Summary should express the core meaning in one concise sentence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "author": author,
                    "transcript": transcript,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=draft_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception:
        return {
            "polished_text": fallback_polished,
            "summary": fallback_summary,
            "warning": "LLM unavailable, returned fallback draft.",
        }

    polished_text, summary, parse_note = _parse_voice_draft_response(raw_text)

    warning = ""
    if not polished_text:
        polished_text = fallback_polished
        warning = "LLM draft parse failed, used transcript directly."
    if not summary:
        summary = await _llm_voice_summary(
            provider=provider,
            transcript=transcript,
            polished_text=polished_text,
            author=author,
        )
    if not summary:
        summary = _fallback_voice_summary(polished_text)
        warning = warning or "LLM summary parse failed, used heuristic summary."

    result = {
        "polished_text": polished_text,
        "summary": summary,
    }
    if warning:
        result["warning"] = warning
    return result


async def _plan_retrieval_route(provider: LLMProvider, user_query: str) -> Dict[str, int]:
    plan = dict(DEFAULT_ROUTE_PLAN)
    if not user_query.strip():
        return plan

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval router for an agentic RAG pipeline. "
                "Return strict JSON only, no markdown, no prose. "
                "Schema: {\"top_k\": int, \"max_sections_per_file\": int, \"min_file_paths\": int}. "
                "Guidelines: list/comparison/coverage-heavy questions should use larger top_k and min_file_paths; "
                "factoid questions use smaller values."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=router_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += chunk.get("content", "")
    except Exception:
        return plan

    parsed = _extract_json_object(raw_text)
    plan["top_k"] = _coerce_int(parsed.get("top_k"), plan["top_k"], 3, 10)
    plan["max_sections_per_file"] = _coerce_int(
        parsed.get("max_sections_per_file"),
        plan["max_sections_per_file"],
        1,
        4,
    )
    plan["min_file_paths"] = _coerce_int(parsed.get("min_file_paths"), plan["min_file_paths"], 2, 8)
    return plan

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check"""
    providers = settings.list_available_providers()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "providers": providers
    }

@app.get("/config")
async def get_config():
    """Get current configuration - dynamically return the current provider's model"""
    config = settings.get_provider_config(settings.api_provider)
    
    return {
        "default_provider": settings.api_provider,
        "default_model": config.get("model", "")
    }

@app.get("/api/config")
async def get_env_config():
    """Read the original content of .env file"""
    env_path = find_dotenv()
    if not env_path:
        raise HTTPException(status_code=404, detail=".env file not found")
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {"content": content}

@app.post("/api/config")
async def update_env_config(request: dict):
    """Directly save .env file content"""
    env_path = find_dotenv()
    if not env_path:
        raise HTTPException(status_code=404, detail=".env file not found")
    
    content = request.get("content", "")
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Config content cannot be empty")
    
    try:
        # Write content directly
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Reload environment variables
        load_dotenv(override=True)
        
        # Reinitialize settings object
        global settings
        from backend.config import Settings
        settings = Settings()
        
        return {"status": "success", "message": "Configuration updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/info", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info():
    try:
        summary = await knowledge_base.get_file_summary()
        file_tree = knowledge_base.list_files()
        return {
            "summary": summary,
            "file_tree": file_tree
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-prompt")
async def get_system_prompt():
    """Return the system prompt currently in use"""
    try:
        file_summary = await knowledge_base.get_file_summary()
        
        # Return corresponding system prompt based on configuration
        if settings.tool_calling_mode == "react":
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        return {
            "system_prompt": system_prompt,
            "mode": settings.tool_calling_mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/retrieve", response_model=FileRetrievalResponse)
async def retrieve_files(request: FileRetrievalRequest):
    try:
        content = await knowledge_base.retrieve_files(request.file_paths)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/draft", response_model=VoiceDraftResponse)
async def draft_voice_note(request: VoiceDraftRequest):
    transcript = _normalize_whitespace(request.transcript)
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript cannot be empty")

    author = _normalize_whitespace(request.author or "Unknown")
    provider = LLMProvider(provider=request.provider or settings.api_provider)
    draft = await _llm_voice_draft(provider=provider, transcript=transcript, author=author)

    return {
        "polished_text": draft.get("polished_text", transcript),
        "summary": draft.get("summary", _fallback_voice_summary(transcript)),
        "warning": draft.get("warning"),
    }


@app.post("/voice/ingest", response_model=VoiceIngestResponse)
async def ingest_voice_note(request: VoiceIngestRequest):
    transcript = (request.transcript or "").strip()
    summary = (request.summary or "").strip()
    author = (request.author or "Unknown").strip()
    source = (request.source or "Realtime voice input").strip()

    if not transcript:
        raise HTTPException(status_code=400, detail="transcript cannot be empty")
    if not summary:
        raise HTTPException(status_code=400, detail="summary cannot be empty")

    occurred_at = _to_local_iso(request.occurred_at)
    dt = datetime.fromisoformat(occurred_at)

    voice_dir = Path(settings.knowledge_base_chunks) / VOICE_NOTES_DIR / dt.strftime("%Y-%m-%d")
    voice_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify_filename(author, fallback="speaker")
    base_name = f"{dt.strftime('%H%M%S')}_{slug}"
    target_file = voice_dir / f"{base_name}.md"
    suffix = 2
    while target_file.exists():
        target_file = voice_dir / f"{base_name}_{suffix}.md"
        suffix += 1

    raw_transcript = (request.raw_transcript or "").strip()
    lines = [
        "# Voice Note",
        "",
        f"- Timestamp: `{occurred_at}`",
        f"- Author: `{author}`",
        f"- Source: `{source}`",
        f"- Summary: {summary}",
        "",
        "## Corrected Transcript",
        transcript,
    ]
    if raw_transcript:
        lines.extend(["", "## Raw Transcript", raw_transcript])

    content = "\n".join(lines).strip() + "\n"
    target_file.write_text(content, encoding="utf-8")

    rel_path = str(target_file.relative_to(Path(settings.knowledge_base_chunks))).replace("\\", "/")
    try:
        transcript_excerpt = _normalize_whitespace(transcript)[:240]
        summary_index_text = (
            f"{summary} | author:{author} | source:{source} | date:{dt.strftime('%Y-%m-%d')} "
            f"| content:{transcript_excerpt}"
        )
        knowledge_base.update_summary_entry(rel_path, summary_index_text)
    except Exception as e:
        print(f"[WARN] Failed to update summary index for {rel_path}: {e}")

    return {
        "status": "ok",
        "file_path": rel_path,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


@app.post("/image/draft", response_model=ImageDraftResponse)
async def draft_image_note(
    file: UploadFile = File(...),
    author: str = Form(default="Unknown"),
    source: str = Form(default="Image upload"),
    provider: Optional[str] = Form(default=None),
):
    author_text = _normalize_whitespace(author or "Unknown")
    source_text = _normalize_whitespace(source or "Image upload")

    image_bytes = await _read_upload_image_bytes(file)
    try:
        width, height, image_format = inspect_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_content_type = (file.content_type or "").strip() or f"image/{image_format.lower()}"
    image_suffix = _guess_image_suffix(file.filename or "", image_content_type, image_format)

    try:
        ocr_result = await run_paddle_ocr(image_bytes=image_bytes, suffix=image_suffix)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    provider_client = LLMProvider(provider=provider or settings.api_provider)
    try:
        vision_result = await run_vision_analysis(
            provider=provider_client,
            image_bytes=image_bytes,
            content_type=image_content_type,
            ocr_text=ocr_result.get("ocr_text", ""),
            author=author_text,
            source=source_text,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "ocr_text": ocr_result.get("ocr_text", ""),
        "visual_summary": vision_result.get("visual_summary", ""),
        "visual_description": vision_result.get("visual_description", ""),
        "tags": vision_result.get("tags", []),
        "retrieval_keywords": vision_result.get("retrieval_keywords", []),
        "ocr_line_count": int(ocr_result.get("ocr_line_count", 0)),
        "image_width": width,
        "image_height": height,
    }


@app.post("/image/ingest", response_model=ImageIngestResponse)
async def ingest_image_note(
    file: UploadFile = File(...),
    visual_summary: str = Form(...),
    visual_description: str = Form(...),
    ocr_text: str = Form(default=""),
    tags: str = Form(default=""),
    retrieval_keywords: str = Form(default=""),
    author: str = Form(default="Unknown"),
    source: str = Form(default="Image upload"),
    occurred_at: Optional[str] = Form(default=None),
):
    summary_text = (visual_summary or "").strip()
    description_text = (visual_description or "").strip()
    ocr_text_value = (ocr_text or "").strip()
    author_text = (author or "Unknown").strip()
    source_text = (source or "Image upload").strip()
    tags_list = parse_tags_text(tags)
    retrieval_keywords_list = parse_tags_text(retrieval_keywords)
    if not retrieval_keywords_list:
        retrieval_keywords_list = tags_list[:]

    if not summary_text:
        raise HTTPException(status_code=400, detail="visual_summary cannot be empty")
    if not description_text:
        raise HTTPException(status_code=400, detail="visual_description cannot be empty")

    image_bytes = await _read_upload_image_bytes(file)
    try:
        width, height, image_format = inspect_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_content_type = (file.content_type or "").strip() or f"image/{image_format.lower()}"
    image_suffix = _guess_image_suffix(file.filename or "", image_content_type, image_format)

    occurred_local = _to_local_iso(occurred_at)
    dt = datetime.fromisoformat(occurred_local)

    kb_root = Path(settings.knowledge_base_chunks)
    day_dir = kb_root / IMAGE_NOTES_DIR / dt.strftime("%Y-%m-%d")
    assets_dir = day_dir / IMAGE_ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify_filename(author_text, fallback="image")
    base_name = f"{dt.strftime('%H%M%S')}_{slug}"
    note_path = day_dir / f"{base_name}.md"
    image_path = assets_dir / f"{base_name}{image_suffix}"
    suffix_counter = 2
    while note_path.exists() or image_path.exists():
        note_path = day_dir / f"{base_name}_{suffix_counter}.md"
        image_path = assets_dir / f"{base_name}_{suffix_counter}{image_suffix}"
        suffix_counter += 1

    image_path.write_bytes(image_bytes)
    rel_image_path = str(image_path.relative_to(kb_root)).replace("\\", "/")

    tags_text = ", ".join(tags_list) if tags_list else "(none)"
    retrieval_keywords_text = (
        ", ".join(retrieval_keywords_list) if retrieval_keywords_list else "(none)"
    )
    lines = [
        "# Image Note",
        "",
        f"- Timestamp: `{occurred_local}`",
        f"- Author: `{author_text}`",
        f"- Source: `{source_text}`",
        f"- Image: `{rel_image_path}`",
        f"- Image Size: `{width}x{height}`",
        f"- Summary: {summary_text}",
        f"- Tags: {tags_text}",
        f"- Retrieval Keywords: {retrieval_keywords_text}",
        "",
        "## Visual Description",
        description_text,
        "",
        "## OCR Text",
        ocr_text_value or "(No text detected by OCR.)",
    ]
    note_content = "\n".join(lines).strip() + "\n"
    note_path.write_text(note_content, encoding="utf-8")

    rel_note_path = str(note_path.relative_to(kb_root)).replace("\\", "/")
    try:
        ocr_excerpt = _normalize_whitespace(ocr_text_value)[:260]
        desc_excerpt = _normalize_whitespace(description_text)[:260]
        summary_index_text = (
            f"{summary_text} | tags:{tags_text} | author:{author_text} | source:{source_text} "
            f"| retrieval_keywords:{retrieval_keywords_text} | date:{dt.strftime('%Y-%m-%d')} "
            f"| image:{rel_image_path} | desc:{desc_excerpt} "
            f"| ocr:{ocr_excerpt}"
        )
        knowledge_base.update_summary_entry(rel_note_path, summary_index_text)
    except Exception as e:
        print(f"[WARN] Failed to update summary index for {rel_note_path}: {e}")

    return {
        "status": "ok",
        "file_path": rel_note_path,
        "image_path": rel_image_path,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        provider = LLMProvider(provider=request.provider or settings.api_provider)
        
        file_summary = await knowledge_base.get_file_summary()
        
        # Check whether to use function calling or ReAct mode
        use_react = settings.tool_calling_mode == "react"
        
        if use_react:
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([msg.dict() for msg in request.messages])
        latest_user_query = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            "",
        )
        query_plan = await _llm_expand_retrieval_query(provider, latest_user_query)
        retrieval_query = query_plan.get("expanded_query") or latest_user_query
        hints = query_plan.get("hints") or []
        if isinstance(hints, list) and hints:
            retrieval_query = _normalize_whitespace(
                f"{retrieval_query} {' '.join(str(h) for h in hints[:10])}"
            )
        route_plan = await _plan_retrieval_route(provider, latest_user_query)
        if query_plan.get("image_intent"):
            route_plan["top_k"] = max(route_plan["top_k"], 8)
            route_plan["min_file_paths"] = max(route_plan["min_file_paths"], 5)
            route_plan["max_sections_per_file"] = max(route_plan["max_sections_per_file"], 2)
        
        if use_react:
            async def generate_response() -> AsyncIterator[str]:
                async for chunk in handle_react_mode(
                    provider,
                    messages,
                    user_query=retrieval_query,
                    route_plan=route_plan,
                ):
                    yield chunk
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            return StreamingResponse(
                generate_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        tools = [
            create_search_paths_tool(),
            create_retrieve_sections_tool(),
        ]
        
        async def generate_response() -> AsyncIterator[str]:
            conversation_messages = messages.copy()
            max_iterations = 12
            iteration = 0
            allowed_tool_names = {"search_paths", "retrieve_sections"}
            has_search_step = False
            candidate_paths: List[str] = []
            evidence_pool: List[Dict[str, str]] = []
            forced_retrieval_attempted = False
            final_answer_sent = False
            
            while iteration < max_iterations:
                iteration += 1
                accumulated_tool_call = None
                iteration_content_buffer = ""
                
                async for chunk_str in provider.chat_completion(
                    messages=conversation_messages,
                    tools=tools,
                    stream=True
                ):
                    try:
                        chunk = json.loads(chunk_str)
                        
                        if chunk["type"] == "content":
                            iteration_content_buffer += chunk.get("content", "")
                        
                        elif chunk["type"] == "usage":
                            yield f"data: {json.dumps({'type': 'usage', 'usage': chunk.get('usage', {})})}\n\n"
                        
                        elif chunk["type"] == "tool_calls":
                            tool_calls = chunk["tool_calls"]
                            
                            for tool_call in tool_calls:
                                if accumulated_tool_call is None:
                                    if tool_call.get("id") and tool_call.get("type"):
                                        accumulated_tool_call = {
                                            "index": tool_call.get("index", 0),
                                            "id": tool_call["id"],
                                            "type": tool_call["type"],
                                            "function": {
                                                "name": tool_call.get("function", {}).get("name", ""),
                                                "arguments": tool_call.get("function", {}).get("arguments", "")
                                            }
                                        }
                                else:
                                    if "function" in tool_call and "arguments" in tool_call["function"]:
                                        accumulated_tool_call["function"]["arguments"] += tool_call["function"]["arguments"]
                    
                    except json.JSONDecodeError:
                        continue
                
                if accumulated_tool_call:
                    tool_name = accumulated_tool_call.get("function", {}).get("name", "")
                    if tool_name not in allowed_tool_names:
                        accumulated_tool_call["function"]["name"] = "search_paths"
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": retrieval_query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )
                        tool_name = "search_paths"

                    # Force retrieval to start from search_paths so retrieve_sections
                    # always receives routed candidate paths first.
                    if tool_name == "retrieve_sections" and not has_search_step:
                        accumulated_tool_call["function"]["name"] = "search_paths"
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": retrieval_query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )
                        tool_name = "search_paths"

                    if tool_name == "search_paths":
                        args = _safe_parse_tool_args(accumulated_tool_call["function"].get("arguments", ""))
                        query = args.get("query") or retrieval_query
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )

                    if tool_name == "retrieve_sections":
                        args = _safe_parse_tool_args(accumulated_tool_call["function"].get("arguments", ""))
                        file_paths = args.get("file_paths") or []
                        if not isinstance(file_paths, list):
                            file_paths = []

                        # Expand with routed candidates to improve coverage without full-file reads.
                        if len(file_paths) < route_plan["min_file_paths"] and candidate_paths:
                            existing = {str(p) for p in file_paths}
                            for p in candidate_paths:
                                if p in existing:
                                    continue
                                file_paths.append(p)
                                existing.add(p)
                                if len(file_paths) >= route_plan["min_file_paths"]:
                                    break

                        max_sections = _coerce_int(
                            args.get("max_sections_per_file"),
                            route_plan["max_sections_per_file"],
                            1,
                            4,
                        )
                        max_sections = max(max_sections, route_plan["max_sections_per_file"])
                        query = args.get("query") or retrieval_query
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "file_paths": file_paths,
                                "query": query or retrieval_query,
                                "max_sections_per_file": max_sections,
                            },
                            ensure_ascii=False,
                        )

                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [accumulated_tool_call]})}\n\n"
                    
                    tool_results = await process_tool_calls([accumulated_tool_call])
                    if tool_name == "search_paths":
                        has_search_step = True
                        candidate_paths = _extract_candidate_paths(tool_results[0].get("content", ""))
                    elif tool_name == "retrieve_sections":
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(tool_results[0].get("content", "")),
                        )
                    
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': tool_results})}\n\n"
                    
                    conversation_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [accumulated_tool_call]
                    })
                    
                    conversation_messages.extend(tool_results)
                    continue

                candidate_answer = iteration_content_buffer.strip()
                if not candidate_answer:
                    break

                # Evidence hard constraint:
                # if no evidence has been retrieved yet, force one more retrieval round.
                if not evidence_pool and not forced_retrieval_attempted:
                    forced_retrieval_attempted = True
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "retrieval_judge",
                                "stop": False,
                                "reason": "证据不足，自动继续检索",
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                    auto_search_call = _build_tool_call(
                        call_id=f"auto_search_{iteration}",
                        name="search_paths",
                        arguments={
                            "query": retrieval_query,
                            "top_k": min(route_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10),
                        },
                    )
                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_search_call]})}\n\n"
                    auto_search_results = await process_tool_calls([auto_search_call])
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_search_results})}\n\n"
                    conversation_messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [auto_search_call]}
                    )
                    conversation_messages.extend(auto_search_results)

                    auto_paths = _extract_candidate_paths(auto_search_results[0].get("content", ""))
                    if auto_paths:
                        candidate_paths = auto_paths
                    has_search_step = True

                    retrieve_targets = candidate_paths[: max(route_plan["min_file_paths"], 2)]
                    if retrieve_targets:
                        auto_retrieve_call = _build_tool_call(
                            call_id=f"auto_retrieve_{iteration}",
                            name="retrieve_sections",
                            arguments={
                                "file_paths": retrieve_targets,
                                "query": retrieval_query,
                                "max_sections_per_file": min(
                                    route_plan["max_sections_per_file"] + 1,
                                    4,
                                ),
                            },
                        )
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_retrieve_call]})}\n\n"
                        auto_retrieve_results = await process_tool_calls([auto_retrieve_call])
                        yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_retrieve_results})}\n\n"
                        conversation_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [auto_retrieve_call],
                            }
                        )
                        conversation_messages.extend(auto_retrieve_results)
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(auto_retrieve_results[0].get("content", "")),
                        )

                    conversation_messages.append({"role": "assistant", "content": candidate_answer})
                    conversation_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous answer did not include evidence citations. "
                                "Re-answer strictly with a final section titled '### 证据' "
                                "and include source file path plus evidence snippet."
                            ),
                        }
                    )
                    continue

                final_answer = _format_answer_with_evidence(candidate_answer, evidence_pool)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "retrieval_judge",
                            "stop": bool(evidence_pool),
                            "reason": (
                                f"证据池通过，共{len(evidence_pool)}条证据片段"
                                if evidence_pool
                                else "证据池未通过"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\n\n"
                final_answer_sent = True
                break
            
            if not final_answer_sent:
                fallback_answer = _format_answer_with_evidence("", evidence_pool)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "retrieval_judge",
                            "stop": False,
                            "reason": "证据不足，未通过证据池校验",
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                yield f"data: {json.dumps({'type': 'content', 'content': fallback_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def list_providers():
    """List all configured providers - use dynamic scanning"""
    provider_ids = settings.list_available_providers()
    
    providers = []
    for provider_id in provider_ids:
        config = settings.get_provider_config(provider_id)
        if config.get("model"):
            providers.append({
                "id": provider_id,
                "name": provider_id.replace('_', ' ').title(),
                "models": [config["model"]]
            })
    
    return {"providers": providers}


@app.get("/evaluation/datasets")
async def list_evaluation_datasets():
    """List available evaluation datasets."""
    try:
        return {"datasets": evaluation_manager.list_datasets()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/summaries")
async def list_evaluation_summaries():
    """List historical evaluation summaries."""
    try:
        return {"summaries": evaluation_manager.list_summaries()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/jobs")
async def list_evaluation_jobs():
    """List in-memory evaluation jobs for this backend process."""
    try:
        return {"jobs": evaluation_manager.list_jobs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/jobs/{job_id}")
async def get_evaluation_job(job_id: str):
    """Get an evaluation job status."""
    try:
        job = evaluation_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluation/jobs/start")
async def start_evaluation_job(request: EvalStartRequest):
    """Start an asynchronous evaluation job."""
    try:
        return evaluation_manager.start_job(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
