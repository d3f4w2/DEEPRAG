from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str


class ChatBudgetConfig(BaseModel):
    max_total_tokens: Optional[int] = None
    max_latency_ms: Optional[int] = None
    price_per_1m_tokens: float = 0.0
    cost_multiplier: float = 1.0


class ChatRequest(BaseModel):
    messages: List[Message]
    provider: Optional[str] = None
    model: Optional[str] = None
    budget: Optional[ChatBudgetConfig] = None

class FileRetrievalRequest(BaseModel):
    file_paths: List[str] = Field(
        ...,
        json_schema_extra={
            "example": [
                "Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md",
                "2023-Market-Layout/",
            ]
        },
    )

class FileRetrievalResponse(BaseModel):
    content: str

class KnowledgeBaseInfo(BaseModel):
    summary: str
    file_tree: Dict[str, Any]

class ProviderConfig(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    providers: List[str]


class EvalStartRequest(BaseModel):
    dataset_path: str
    run_name: Optional[str] = None
    provider: Optional[str] = None
    compare_summary_path: Optional[str] = None
    max_questions: int = 0
    timeout_sec: float = 180.0
    generate_analysis: bool = True
    base_url: Optional[str] = None


class VoiceDraftRequest(BaseModel):
    transcript: str
    author: Optional[str] = None
    provider: Optional[str] = None


class VoiceDraftResponse(BaseModel):
    polished_text: str
    summary: str
    warning: Optional[str] = None


class VoiceIngestRequest(BaseModel):
    transcript: str
    summary: str
    author: str = Field(default="Unknown")
    source: str = Field(default="Realtime voice input")
    occurred_at: Optional[str] = None
    raw_transcript: Optional[str] = None


class VoiceIngestResponse(BaseModel):
    status: str
    file_path: str
    created_at: str


class ImageDraftResponse(BaseModel):
    ocr_text: str
    visual_summary: str
    visual_description: str
    tags: List[str]
    retrieval_keywords: List[str]
    ocr_line_count: int
    image_width: int
    image_height: int


class ImageIngestResponse(BaseModel):
    status: str
    file_path: str
    image_path: str
    created_at: str
