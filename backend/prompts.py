import json
from typing import Dict, List
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


async def process_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("function", {}).get("name")
        try:
            args = _parse_tool_arguments(tool_call["function"]["arguments"])

            if tool_name == "search_paths":
                query = args.get("query", "")
                top_k = args.get("top_k", 5)
                content = await knowledge_base.search_paths(query=query, top_k=top_k)
            elif tool_name == "retrieve_sections":
                file_paths = args.get("file_paths", [])
                query = args.get("query", "")
                max_sections = args.get("max_sections_per_file", 2)
                content = await knowledge_base.retrieve_sections(
                    file_paths=file_paths,
                    query=query,
                    max_sections_per_file=max_sections,
                )
            elif tool_name == "retrieve_files":
                content = (
                    "Tool `retrieve_files` is disabled in this agentic mode. "
                    "Use `search_paths` first, then `retrieve_sections`."
                )
            else:
                content = f"Unsupported tool: {tool_name}"

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": content
            })
        except Exception as e:
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": f"Error executing {tool_name}: {str(e)}"
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

    if has_action and action == "retrieve_files":
        file_paths = action_input.get("file_paths", [])
        content = await knowledge_base.retrieve_files(file_paths)

        return {
            "action": action,
            "input": action_input,
            "observation": content
        }, True

    if has_action and action == "search_paths":
        query = action_input.get("query", "")
        top_k = action_input.get("top_k", 5)
        content = await knowledge_base.search_paths(query=query, top_k=top_k)

        return {
            "action": action,
            "input": action_input,
            "observation": content
        }, True

    if has_action and action == "retrieve_sections":
        file_paths = action_input.get("file_paths", [])
        query = action_input.get("query", "")
        max_sections = action_input.get("max_sections_per_file", 2)
        content = await knowledge_base.retrieve_sections(
            file_paths=file_paths,
            query=query,
            max_sections_per_file=max_sections,
        )

        return {
            "action": action,
            "input": action_input,
            "observation": content
        }, True

    return None, False
