import asyncio
import json
import os
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from datetime import datetime
from backend.knowledge_base import knowledge_base


def _create_base_system_prompt(file_summary: str) -> str:
    """Shared base system prompt"""
    current_time = datetime.now().strftime("%A, %B %d, %Y, at %I:%M:%S %p")
    
    return f"""
- Answers must strictly come from the knowledge base
- Use an evidence-first workflow:
  1) `search_paths` to find likely files
  2) `retrieve_sections` to extract relevant evidence snippets
- Use only available tools: `search_paths` and `retrieve_sections`
- Final answer MUST include a section titled `### 证据`
- In `### 证据`, each item must contain:
  - `来源文件: <path>`
  - `证据片段: "<snippet>"`
- If you're 100% certain from the file summary, you may answer directly, but still include `### 证据`
- If evidence is still insufficient after focused retrieval, answer "I don't know"
- Current time: {current_time}

## Knowledge Base File Summary
```
{file_summary}
```

## search_paths
- Input format: {{"query": "user question", "top_k": 5}}
- Output: candidate file paths with relevance scores

## retrieve_sections
- Input format: {{"file_paths": ["path1", "path2"], "query": "user question", "max_sections_per_file": 2}}
- Output: high-relevance sections from candidate files

## Token control rules
- Prefer section retrieval over full-file retrieval
- Keep retrieval focused and iterative
- If evidence is insufficient, rerun `search_paths` with broader query and call `retrieve_sections` again

""".strip()


def create_system_prompt(file_summary: str) -> str:
    """System prompt for function calling mode"""
    return _create_base_system_prompt(file_summary)


def create_react_system_prompt(file_summary: str) -> str:
    """System prompt for ReAct mode with format instructions"""
    base_prompt = _create_base_system_prompt(file_summary)
    
    return f"""
{base_prompt}

## Direct Answer
- `Knowledge Base File Summary` has the answer

### Example
- Question: Besides AMOLED and OLED screens, what other display types do we have?
- Answer: LCD, TFT

## Tool Call
- `Knowledge Base File Summary` doesn't have enough details

### Pattern
- <|Thought|> Think about what information you need to answer the question
- <|Action|> Tool
- <|Action Input|> Input format
- <|Observation|> [The system will provide file contents here]
- ... (repeat Thought/Action/Observation as needed)
- <|Final Answer|> [Your final answer based on the retrieved information]

### Example
- Question: What are all the technical specifications of SW-2100?
- <|Thought|> I need targeted evidence snippets from likely files first
- <|Action|> search_paths
- <|Action Input|> {{"query": "technical specifications of SW-2100", "top_k": 6}}
- <|Observation|> [System provides candidate file paths]
- <|Action|> retrieve_sections
- <|Action Input|> {{"file_paths": ["Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md"], "query": "technical specifications of SW-2100", "max_sections_per_file": 3}}
- <|Observation|> [System provides evidence snippets]
- <|Final Answer|> [Answer + ### 证据 with source files and snippets]

""".strip()
    

def create_file_retrieval_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_files",
            "description": "Fallback tool: retrieve full file contents from knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths or directory paths to retrieve."
                    }
                },
                "required": ["file_paths"]
            }
        }
    }


def create_search_paths_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "search_paths",
            "description": "Find the most relevant file paths from the knowledge map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User question or retrieval intent."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of candidate paths to return.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }


def create_retrieve_sections_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_sections",
            "description": "Retrieve only the most relevant sections from candidate files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate file paths from search_paths."
                    },
                    "query": {
                        "type": "string",
                        "description": "User question for section relevance scoring."
                    },
                    "max_sections_per_file": {
                        "type": "integer",
                        "description": "Maximum sections extracted from each file.",
                        "default": 2
                    }
                },
                "required": ["file_paths", "query"]
            }
        }
    }


def _parse_tool_arguments(arguments: str) -> Dict:
    """Parse tool arguments robustly when model output includes extra trailing text."""
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        cleaned = (arguments or "").strip()
        decoder = json.JSONDecoder()

        try:
            parsed, _ = decoder.raw_decode(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])

        raise


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed != parsed:
        parsed = default
    return max(min_value, min(parsed, max_value))


TOOL_CACHE_TTL_SECONDS = _clamp_float(
    os.getenv("TOOL_CACHE_TTL_SECONDS"),
    default=300.0,
    min_value=0.0,
    max_value=86_400.0,
)
TOOL_CACHE_MAX_ENTRIES = _clamp_int(
    os.getenv("TOOL_CACHE_MAX_ENTRIES"),
    default=256,
    min_value=0,
    max_value=10_000,
)
TOOL_RETRY_MAX_ATTEMPTS = _clamp_int(
    os.getenv("TOOL_RETRY_MAX_ATTEMPTS"),
    default=3,
    min_value=1,
    max_value=8,
)
TOOL_RETRY_BASE_DELAY_SECONDS = _clamp_float(
    os.getenv("TOOL_RETRY_BASE_DELAY_SECONDS"),
    default=0.35,
    min_value=0.0,
    max_value=10.0,
)
TOOL_RETRY_MAX_DELAY_SECONDS = _clamp_float(
    os.getenv("TOOL_RETRY_MAX_DELAY_SECONDS"),
    default=3.0,
    min_value=0.0,
    max_value=30.0,
)

_CACHEABLE_TOOL_NAMES = {"search_paths", "retrieve_sections"}
_RETRYABLE_ERROR_KEYWORDS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "temporary failure",
    "reset by peer",
    "network",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
)

# LRU + TTL cache for tool outputs to reduce repeated retrieval overhead.
_tool_cache: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()


def _normalize_tool_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_tool_args(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_normalize_tool_args(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_tool_args(item) for item in value]
    return value


def _build_tool_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    normalized = _normalize_tool_args(args or {})
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{tool_name}:{payload}"


def _cache_get(cache_key: str) -> str | None:
    if TOOL_CACHE_MAX_ENTRIES <= 0 or TOOL_CACHE_TTL_SECONDS <= 0:
        return None
    cached = _tool_cache.get(cache_key)
    if not cached:
        return None

    expires_at, content = cached
    if expires_at <= time.time():
        _tool_cache.pop(cache_key, None)
        return None

    _tool_cache.move_to_end(cache_key)
    return content


def _cache_set(cache_key: str, content: str) -> None:
    if TOOL_CACHE_MAX_ENTRIES <= 0 or TOOL_CACHE_TTL_SECONDS <= 0:
        return

    expires_at = time.time() + TOOL_CACHE_TTL_SECONDS
    _tool_cache[cache_key] = (expires_at, content)
    _tool_cache.move_to_end(cache_key)

    while len(_tool_cache) > TOOL_CACHE_MAX_ENTRIES:
        _tool_cache.popitem(last=False)


def _is_retryable_exception(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in _RETRYABLE_ERROR_KEYWORDS)


def _backoff_delay_seconds(attempt_index: int) -> float:
    # Exponential backoff with jitter to avoid synchronized retries.
    if TOOL_RETRY_BASE_DELAY_SECONDS <= 0:
        return 0.0
    exponential = TOOL_RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempt_index - 1))
    capped = min(exponential, TOOL_RETRY_MAX_DELAY_SECONDS)
    jitter = capped * random.uniform(0.0, 0.25)
    return max(0.0, capped + jitter)


async def _run_tool_once(tool_name: str, args: Dict[str, Any]) -> str:
    if tool_name == "search_paths":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        return await knowledge_base.search_paths(query=query, top_k=top_k)
    if tool_name == "retrieve_sections":
        file_paths = args.get("file_paths", [])
        query = args.get("query", "")
        max_sections = args.get("max_sections_per_file", 2)
        return await knowledge_base.retrieve_sections(
            file_paths=file_paths,
            query=query,
            max_sections_per_file=max_sections,
        )
    if tool_name == "retrieve_files":
        return (
            "Tool `retrieve_files` is disabled in this agentic mode. "
            "Use `search_paths` first, then `retrieve_sections`."
        )
    return f"Unsupported tool: {tool_name}"


async def _run_tool_with_retry(
    tool_name: str,
    args: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    attempts = 0
    retry_count = 0
    last_exc: Exception | None = None

    for attempt in range(1, TOOL_RETRY_MAX_ATTEMPTS + 1):
        attempts = attempt
        try:
            content = await _run_tool_once(tool_name, args)
            return content, {
                "attempts": attempts,
                "retry_count": retry_count,
                "retry_exhausted": False,
            }
        except Exception as exc:
            last_exc = exc
            if attempt >= TOOL_RETRY_MAX_ATTEMPTS or not _is_retryable_exception(exc):
                break
            retry_count += 1
            delay = _backoff_delay_seconds(attempt)
            if delay > 0:
                await asyncio.sleep(delay)

    error_text = str(last_exc) if last_exc else "unknown tool error"
    return (
        f"Error executing {tool_name}: {error_text}",
        {
            "attempts": attempts,
            "retry_count": retry_count,
            "retry_exhausted": True,
        },
    )


async def process_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("function", {}).get("name")
        try:
            args = _parse_tool_arguments(tool_call["function"]["arguments"])
            cache_key = ""
            cache_hit = False
            retry_meta: Dict[str, Any] = {
                "attempts": 0,
                "retry_count": 0,
                "retry_exhausted": False,
            }

            if tool_name in _CACHEABLE_TOOL_NAMES:
                cache_key = _build_tool_cache_key(tool_name, args)
                cached = _cache_get(cache_key)
                if cached is not None:
                    content = cached
                    cache_hit = True
                else:
                    content, retry_meta = await _run_tool_with_retry(tool_name, args)
                    if not retry_meta.get("retry_exhausted", False):
                        _cache_set(cache_key, content)
            else:
                content, retry_meta = await _run_tool_with_retry(tool_name, args)

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": content,
                "meta": {
                    "tool_name": tool_name,
                    "cache_hit": cache_hit,
                    "attempts": int(retry_meta.get("attempts", 0) or 0),
                    "retry_count": int(retry_meta.get("retry_count", 0) or 0),
                    "retry_exhausted": bool(retry_meta.get("retry_exhausted", False)),
                },
            })
        except Exception as e:
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": f"Error executing {tool_name}: {str(e)}",
                "meta": {
                    "tool_name": tool_name,
                    "cache_hit": False,
                    "attempts": 0,
                    "retry_count": 0,
                    "retry_exhausted": True,
                },
            })

    return results


def parse_react_response(text: str) -> tuple:
    """Parse ReAct-style response to extract action and input"""
    import re

    # 查找 <|Action|> 和 <|Action Input|> (新格式)
    action_pattern = r'<\|Action\|>\s*(\w+)'
    action_input_pattern = r'<\|Action Input\|>\s*(\{[^}]+\})'

    action_match = re.search(action_pattern, text)
    action_input_match = re.search(action_input_pattern, text)

    if action_match and action_input_match:
        action = action_match.group(1)
        try:
            action_input = json.loads(action_input_match.group(1))
            return action, action_input, True
        except:
            pass

    return None, None, False


async def process_react_response(text: str) -> tuple:
    """Process ReAct response and execute actions"""
    action, action_input, has_action = parse_react_response(text)
    if has_action and action in {"retrieve_files", "search_paths", "retrieve_sections"}:
        synthetic_tool_call = {
            "id": "react_call",
            "type": "function",
            "function": {
                "name": action,
                "arguments": json.dumps(action_input or {}, ensure_ascii=False),
            },
        }
        tool_results = await process_tool_calls([synthetic_tool_call])
        first_result = tool_results[0] if tool_results else {}
        content = str(first_result.get("content", ""))
        meta = first_result.get("meta", {})

        return {
            "action": action,
            "input": action_input,
            "observation": content,
            "meta": meta,
        }, True

    return None, False
