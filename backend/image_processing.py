import asyncio
import base64
import io
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from backend.llm_provider import LLMProvider


MAX_VISION_OCR_CHARS = int(os.getenv("MAX_VISION_OCR_CHARS", "5000"))


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def parse_tags_text(raw: str) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []

    parsed: Any = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        result: List[str] = []
        for item in parsed:
            value = normalize_text(str(item))
            if value and value not in result:
                result.append(value)
        return result

    result: List[str] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        value = normalize_text(token)
        if value and value not in result:
            result.append(value)
    return result


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_tags(raw_tags: Any) -> List[str]:
    if isinstance(raw_tags, list):
        values = [normalize_text(str(item)) for item in raw_tags]
        return [tag for idx, tag in enumerate(values) if tag and tag not in values[:idx]]
    if isinstance(raw_tags, str):
        return parse_tags_text(raw_tags)
    return []


def inspect_image_bytes(image_bytes: bytes) -> Tuple[int, int, str]:
    if not image_bytes:
        raise ValueError("Image file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width = int(image.width)
            height = int(image.height)
            image_format = (image.format or "PNG").upper()
    except UnidentifiedImageError as e:
        raise ValueError("Unsupported image file or invalid image content.") from e
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size.")
    return width, height, image_format


class PaddleOCRRuntime:
    _engine: Any = None
    _compat_mode_enabled: bool = False

    @classmethod
    def _is_pir_onednn_error(cls, error: Exception) -> bool:
        text = str(error or "").lower()
        return (
            "convertpirattribute2runtimeattribute" in text
            or ("onednn_instruction.cc" in text and "unimplemented" in text)
        )

    @classmethod
    def _enable_compat_mode(cls) -> None:
        cls._compat_mode_enabled = True
        # Workaround for Paddle PIR + oneDNN compatibility issue on some CPU builds.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
        try:
            import paddle  # type: ignore

            for flags in (
                {"FLAGS_use_mkldnn": False},
                {"FLAGS_use_mkldnn": 0},
                {"FLAGS_enable_pir_api": False},
                {"FLAGS_enable_pir_api": 0},
                {"FLAGS_enable_pir_in_executor": False},
                {"FLAGS_enable_pir_in_executor": 0},
            ):
                try:
                    paddle.set_flags(flags)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def _init_engine(cls) -> Any:
        if cls._engine is not None:
            return cls._engine
        cache_root = Path.cwd() / ".paddle_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        temp_root = cache_root / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        if not os.getenv("PADDLE_PDX_CACHE_HOME"):
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_root / "paddlex")
        if not os.getenv("PADDLE_HOME"):
            os.environ["PADDLE_HOME"] = str(cache_root)
        if not os.getenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"):
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["TMP"] = str(temp_root)
        os.environ["TEMP"] = str(temp_root)
        os.environ["TMPDIR"] = str(temp_root)
        os.environ["HOME"] = str(cache_root)
        if os.name == "nt":
            os.environ["USERPROFILE"] = str(cache_root)
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PaddleOCR is not installed. Install in conda env 'rag' with "
                "`pip install paddleocr paddlepaddle opencv-python-headless Pillow`."
            ) from e

        ocr_lang = os.getenv("PADDLEOCR_LANG", "ch")
        try:
            init_params = inspect.signature(PaddleOCR.__init__).parameters
            kwargs: Dict[str, Any] = {"lang": ocr_lang}
            # PaddleOCR 3.x common args (passed through **kwargs):
            # force stable CPU paddle runtime path by default.
            stable_runtime_flags: Dict[str, Any] = {
                "device": os.getenv("PADDLEOCR_DEVICE", "cpu"),
                "enable_mkldnn": os.getenv("PADDLEOCR_ENABLE_MKLDNN", "false").lower() == "true",
                "enable_cinn": os.getenv("PADDLEOCR_ENABLE_CINN", "false").lower() == "true",
                "enable_hpi": os.getenv("PADDLEOCR_ENABLE_HPI", "false").lower() == "true",
                "cpu_threads": int(os.getenv("PADDLEOCR_CPU_THREADS", "4")),
            }
            optional_flags: Dict[str, Any] = {
                # Avoid downloading/using extra preprocessor branches unless needed.
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            kwargs.update(stable_runtime_flags)
            for key, value in optional_flags.items():
                if key in init_params:
                    kwargs[key] = value
            cls._engine = PaddleOCR(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e
        return cls._engine

    @classmethod
    def _invoke_with_fallback(cls, engine: Any, image_path: str) -> Any:
        attempts = [
            lambda: engine.predict(image_path),
            lambda: engine.predict([image_path]),
            lambda: engine.ocr(image_path),
            lambda: engine.ocr(image_path, det=True, rec=True),
            lambda: engine.ocr([image_path]),
        ]
        last_error: Optional[Exception] = None
        for call in attempts:
            try:
                return call()
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"PaddleOCR invocation failed: {last_error}")

    @classmethod
    def run_ocr_sync(cls, image_path: str) -> Any:
        engine = cls._init_engine()
        try:
            return cls._invoke_with_fallback(engine, image_path)
        except RuntimeError as first_error:
            if cls._compat_mode_enabled or not cls._is_pir_onednn_error(first_error):
                raise
            cls._enable_compat_mode()
            cls._engine = None
            retry_engine = cls._init_engine()
            return cls._invoke_with_fallback(retry_engine, image_path)


def _flatten_ocr_result(raw_result: Any) -> List[Tuple[str, Optional[float]]]:
    lines: List[Tuple[str, Optional[float]]] = []
    seen: set[Tuple[str, Optional[float]]] = set()

    def _to_score(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _append(text: Any, score: Any = None):
        normalized = str(text or "").strip()
        if not normalized:
            return
        score_f = _to_score(score)
        key = (normalized, score_f)
        if key in seen:
            return
        seen.add(key)
        lines.append((normalized, score_f))

    def _walk(node: Any):
        if node is None:
            return

        if hasattr(node, "to_dict") and callable(getattr(node, "to_dict")):
            try:
                _walk(node.to_dict())
                return
            except Exception:
                pass
        if hasattr(node, "model_dump") and callable(getattr(node, "model_dump")):
            try:
                _walk(node.model_dump())
                return
            except Exception:
                pass

        if isinstance(node, dict):
            text_keys = ["text", "rec_text", "ocr_text", "transcription"]
            score_keys = ["score", "confidence", "rec_score"]
            for tk in text_keys:
                if tk in node:
                    score_val = None
                    for sk in score_keys:
                        if sk in node:
                            score_val = node.get(sk)
                            break
                    _append(node.get(tk), score_val)

            rec_texts = node.get("rec_texts") or node.get("texts") or node.get("ocr_texts")
            rec_scores = node.get("rec_scores") or node.get("scores")
            if isinstance(rec_texts, list):
                for idx, item in enumerate(rec_texts):
                    score_val = None
                    if isinstance(rec_scores, list) and idx < len(rec_scores):
                        score_val = rec_scores[idx]
                    _append(item, score_val)

            for value in node.values():
                _walk(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                rec = node[1]
                if len(rec) >= 1:
                    score_val = rec[1] if len(rec) >= 2 else None
                    _append(rec[0], score_val)
            elif (
                len(node) >= 2
                and len(node) <= 3
                and isinstance(node[0], str)
                and isinstance(node[1], (int, float, type(None)))
            ):
                score_val = node[1] if len(node) >= 2 else None
                _append(node[0], score_val)

            for item in node:
                _walk(item)
            return

    _walk(raw_result)
    return lines


async def run_paddle_ocr(image_bytes: bytes, suffix: str = ".png") -> Dict[str, Any]:
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        raw = await asyncio.to_thread(PaddleOCRRuntime.run_ocr_sync, str(temp_path))
        lines = _flatten_ocr_result(raw)
        texts = [text for text, _ in lines]
        confidences = [score for _, score in lines if score is not None]
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        return {
            "ocr_text": "\n".join(texts).strip(),
            "ocr_line_count": len(texts),
            "ocr_avg_confidence": avg_conf,
        }
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}") from e
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


async def run_vision_analysis(
    provider: LLMProvider,
    image_bytes: bytes,
    content_type: str,
    ocr_text: str,
    author: str,
    source: str,
) -> Dict[str, Any]:
    safe_content_type = (content_type or "image/png").lower()
    if not safe_content_type.startswith("image/"):
        safe_content_type = "image/png"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{safe_content_type};base64,{b64_image}"
    clipped_ocr = (ocr_text or "").strip()
    if len(clipped_ocr) > MAX_VISION_OCR_CHARS:
        clipped_ocr = clipped_ocr[:MAX_VISION_OCR_CHARS]

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an image analyst for RAG ingestion. "
                "Return strict JSON only, no markdown. "
                "Schema: {"
                "\"visual_summary\": string, "
                "\"visual_description\": string, "
                "\"tags\": string[], "
                "\"retrieval_keywords\": string[]"
                "}. "
                "Requirements: "
                "1) visual_summary and visual_description must be bilingual in one string (Chinese first, then English). "
                "2) tags and retrieval_keywords should include Chinese+English terms for objects, scene, attributes, and intent-level terms. "
                "3) retrieval_keywords should be 8-20 concise search terms suitable for lexical retrieval. "
                "4) Do not fabricate details not visible in the image."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Author: {author or 'Unknown'}\n"
                        f"Source: {source or 'Image upload'}\n"
                        f"OCR_TEXT:\n{clipped_ocr or '(empty)'}\n\n"
                        "Please provide a concise summary, detailed visual description, and 3-8 tags."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        },
    ]

    raw_output = ""
    try:
        async for chunk_str in provider.chat_completion(messages=messages, tools=None, stream=False):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_output += str(chunk.get("content") or "")
    except Exception as e:
        raise RuntimeError(f"Vision model request failed: {e}") from e

    if not raw_output.strip():
        raise RuntimeError("Vision model returned empty response.")
    if raw_output.strip().lower().startswith("error:"):
        raise RuntimeError(raw_output.strip())

    parsed = _extract_json_object(raw_output)
    if not parsed:
        raise RuntimeError("Vision model did not return valid JSON.")

    visual_summary = normalize_text(str(parsed.get("visual_summary") or ""))
    visual_description = normalize_text(str(parsed.get("visual_description") or ""))
    tags = _safe_tags(parsed.get("tags"))
    retrieval_keywords = _safe_tags(parsed.get("retrieval_keywords"))

    if not visual_summary:
        raise RuntimeError("Vision model JSON missing field: visual_summary.")
    if not visual_description:
        raise RuntimeError("Vision model JSON missing field: visual_description.")
    if not retrieval_keywords:
        retrieval_keywords = tags[:]

    return {
        "visual_summary": visual_summary,
        "visual_description": visual_description,
        "tags": tags,
        "retrieval_keywords": retrieval_keywords,
    }
