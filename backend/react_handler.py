from typing import Any, AsyncIterator, Dict, List
import json
import re

from backend.prompts import process_react_response, process_tool_calls

DEFAULT_ROUTE_PLAN = {
    "top_k": 6,
    "max_sections_per_file": 2,
    "min_file_paths": 4,
}
AUTO_RETRIEVAL_TOP_K_BOOST = 2
MAX_AUTO_RETRIEVAL_ROUNDS = 3
MAX_FINAL_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_SNIPPET_LENGTH = 280
FALLBACK_NO_EVIDENCE_ANSWER = (
    "我无法给出有证据支撑的回答。已自动再检索多轮，但仍未找到可引用片段。"
)

EVIDENCE_SECTION_PATTERN = re.compile(r"(?is)\n#{2,3}\s*(证据|evidence)\s*\n.*$")
EVIDENCE_BLOCK_PATTERN = re.compile(
    r"\[\[FILE:(?P<path>.+?)\s*\|\s*SECTION:(?P<section>\d+)\s*\|\s*SCORE:(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*HEADING:(?P<heading>.*?)\]\]\s*(?P<body>.*?)(?=(?:\n\n==========\n\n|\n\n----------\n\n|\Z))",
    re.DOTALL,
)
FINAL_ANSWER_PATTERN = re.compile(r"<\|Final Answer\|>\s*([\s\S]*)", re.IGNORECASE)


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


async def _llm_should_stop_retrieval(
    provider,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    if not evidence_pool:
        return {"stop": False, "reason": "no evidence in pool"}

    evidence_preview = []
    for item in evidence_pool[:4]:
        evidence_preview.append(
            {
                "file_path": item.get("file_path", ""),
                "snippet": item.get("snippet", "")[:180],
            }
        )

    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval stop-judge for RAG. "
                "Return strict JSON only: {\"stop\": bool, \"reason\": string}. "
                "Set stop=true only when evidence is sufficient to support the answer. "
                "If likely incomplete, ambiguous, or missing key facets, set stop=false."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": user_query,
                    "candidate_answer": candidate_answer,
                    "evidence": evidence_preview,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=judge_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += chunk.get("content", "")
    except Exception:
        return {"stop": True, "reason": "judge failed; fallback to evidence-present"}

    parsed = _extract_json_object(raw_text)
    stop = bool(parsed.get("stop"))
    reason = str(parsed.get("reason", "")).strip() or "no reason"
    return {"stop": stop, "reason": reason}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _shorten_text(text: str, max_len: int = 72) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


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


def _extract_final_answer_text(accumulated_response: str) -> str:
    if not accumulated_response:
        return ""

    match = FINAL_ANSWER_PATTERN.search(accumulated_response)
    if match:
        return match.group(1).strip()

    return accumulated_response.strip()


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


async def handle_react_mode(
    provider,
    messages,
    user_query: str = "",
    route_plan: Dict[str, int] | None = None,
    max_iterations: int = 10,
) -> AsyncIterator[str]:
    """Handle ReAct-style tool calling with hard evidence validation and auto-retrieval."""

    conversation_messages = messages.copy()
    iteration = 0

    safe_plan = dict(DEFAULT_ROUTE_PLAN)
    if route_plan:
        safe_plan.update(route_plan)

    evidence_pool: List[Dict[str, str]] = []
    candidate_paths: List[str] = []
    auto_retrieval_rounds = 0
    final_answer_sent = False

    while iteration < max_iterations:
        iteration += 1
        accumulated_response = ""
        action_detected = False
        action_result = None
        has_action = False

        async for chunk_str in provider.chat_completion(
            messages=conversation_messages,
            tools=None,
            stream=True,
        ):
            try:
                chunk = json.loads(chunk_str)

                if chunk["type"] == "content":
                    content = chunk.get("content") or ""
                    accumulated_response += content

                    if "<|Action Input|>" in accumulated_response and "{" in accumulated_response:
                        parsed_result, parsed_has_action = await process_react_response(accumulated_response)
                        if parsed_has_action:
                            action_detected = True
                            action_result = parsed_result
                            has_action = True
                            break

                elif chunk["type"] == "usage":
                    yield f"data: {json.dumps({'type': 'usage', 'usage': chunk.get('usage', {})})}\\n\\n"

            except json.JSONDecodeError:
                continue

        if not action_detected:
            action_result, has_action = await process_react_response(accumulated_response)

        if has_action and action_result:
            tool_call_info = {
                "id": f"call_{iteration}",
                "type": "function",
                "function": {
                    "name": action_result["action"],
                    "arguments": json.dumps(action_result["input"], ensure_ascii=False),
                },
            }
            yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [tool_call_info]})}\\n\\n"

            observation = action_result.get("observation", "")
            content_length = len(observation)
            if content_length > 50000:
                frontend_content = (
                    f"[Retrieved {content_length:,} characters from knowledge base. "
                    "Content sent to LLM for analysis.]"
                )
            else:
                frontend_content = observation

            tool_result_info = {
                "role": "tool",
                "tool_call_id": f"call_{iteration}",
                "content": frontend_content,
            }
            yield f"data: {json.dumps({'type': 'tool_results', 'results': [tool_result_info]})}\\n\\n"

            if action_result["action"] == "search_paths":
                candidate_paths = _extract_candidate_paths(observation)
            elif action_result["action"] == "retrieve_sections":
                evidence_pool = _merge_evidence_pool(
                    evidence_pool,
                    _extract_evidence_entries(observation),
                )

            conversation_messages.append(
                {
                    "role": "assistant",
                    "content": accumulated_response,
                }
            )
            conversation_messages.append(
                {
                    "role": "user",
                    "content": f"<|Observation|> {observation}\\n\\nContinue with your reasoning.",
                }
            )
            continue

        candidate_answer = _extract_final_answer_text(accumulated_response)
        stop_decision = None
        if evidence_pool:
            stop_decision = await _llm_should_stop_retrieval(
                provider=provider,
                user_query=user_query,
                candidate_answer=candidate_answer,
                evidence_pool=evidence_pool,
            )
            judge_payload = {
                "type": "retrieval_judge",
                "stop": bool(stop_decision.get("stop", False)),
                "reason": _shorten_text(str(stop_decision.get("reason", ""))),
            }
            yield f"data: {json.dumps(judge_payload, ensure_ascii=False)}\\n\\n"
            if stop_decision.get("stop", False):
                final_answer = _format_answer_with_evidence(candidate_answer, evidence_pool)
                yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\\n\\n"
                final_answer_sent = True
                break

        if auto_retrieval_rounds >= MAX_AUTO_RETRIEVAL_ROUNDS:
            yield f"data: {json.dumps({'type': 'content', 'content': FALLBACK_NO_EVIDENCE_ANSWER})}\\n\\n"
            final_answer_sent = True
            break

        auto_retrieval_rounds += 1
        retrieval_query = user_query or candidate_answer or "knowledge base question"

        auto_search_call = _build_tool_call(
            call_id=f"auto_search_{iteration}_{auto_retrieval_rounds}",
            name="search_paths",
            arguments={
                "query": retrieval_query,
                "top_k": min(safe_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10),
            },
        )
        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_search_call]})}\\n\\n"
        auto_search_results = await process_tool_calls([auto_search_call])
        yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_search_results})}\\n\\n"
        auto_search_output = auto_search_results[0].get("content", "")
        auto_paths = _extract_candidate_paths(auto_search_output)
        if auto_paths:
            candidate_paths = auto_paths

        retrieve_targets = candidate_paths[: max(int(safe_plan.get("min_file_paths", 4)), 2)]
        auto_retrieve_output = ""
        if retrieve_targets:
            auto_retrieve_call = _build_tool_call(
                call_id=f"auto_retrieve_{iteration}_{auto_retrieval_rounds}",
                name="retrieve_sections",
                arguments={
                    "file_paths": retrieve_targets,
                    "query": retrieval_query,
                    "max_sections_per_file": min(
                        int(safe_plan.get("max_sections_per_file", 2)) + 1,
                        4,
                    ),
                },
            )
            yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_retrieve_call]})}\\n\\n"
            auto_retrieve_results = await process_tool_calls([auto_retrieve_call])
            yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_retrieve_results})}\\n\\n"
            auto_retrieve_output = auto_retrieve_results[0].get("content", "")
            evidence_pool = _merge_evidence_pool(
                evidence_pool,
                _extract_evidence_entries(auto_retrieve_output),
            )

        conversation_messages.append({"role": "assistant", "content": accumulated_response})

        observation_parts = [
            f"<|Observation|> Auto search_paths result:\n{auto_search_output}",
        ]
        if auto_retrieve_output:
            observation_parts.append(
                f"<|Observation|> Auto retrieve_sections result:\n{auto_retrieve_output}"
            )

        if evidence_pool:
            guard_prompt = (
                "Evidence snippets are now available. Decide whether they are sufficient. "
                "If sufficient, provide final answer with `### 证据`; "
                "if not, continue retrieval."
            )
            if isinstance(stop_decision, dict):
                judge_reason = str(stop_decision.get("reason", "")).strip()
                if judge_reason:
                    guard_prompt += f" Stop judge reason: {judge_reason}"
        else:
            guard_prompt = (
                "No evidence snippet found yet. Please continue retrieval with broader/narrower query. "
                "If still insufficient after reasonable attempts, answer I don't know."
            )

        conversation_messages.append(
            {
                "role": "user",
                "content": "\\n\\n".join(observation_parts + [guard_prompt]),
            }
        )

    if not final_answer_sent:
        fallback = _format_answer_with_evidence("", evidence_pool)
        yield f"data: {json.dumps({'type': 'content', 'content': fallback})}\\n\\n"
