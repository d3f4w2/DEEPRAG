import asyncio
import json
import shutil
from pathlib import Path

from backend.knowledge_base import KnowledgeBase


def _make_local_tmp_dir(name: str) -> Path:
    path = Path("tests") / "_tmp" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_search_paths_keeps_lexical_fallback_when_vector_unavailable(monkeypatch):
    temp_dir = _make_local_tmp_dir("lexical_fallback")
    kb = KnowledgeBase(base_path=str(temp_dir))

    monkeypatch.setattr(
        kb,
        "_load_summary_index",
        lambda: [
            {"path": "display_types.md", "summary": "LCD TFT PMOLED display technologies"},
            {"path": "voice_notes.md", "summary": "weekly meeting transcript"},
        ],
    )

    async def fake_vector_scores(entries, query):
        return {}

    monkeypatch.setattr(kb, "_compute_vector_scores", fake_vector_scores)

    result = asyncio.run(kb.search_paths("display types", top_k=2))
    payload = json.loads(result)

    assert payload["candidates"][0]["path"] == "display_types.md"
    assert payload["candidates"][0]["score"] > 0


def test_search_paths_can_return_vector_only_match(monkeypatch):
    temp_dir = _make_local_tmp_dir("vector_only_match")
    kb = KnowledgeBase(base_path=str(temp_dir))

    monkeypatch.setattr(
        kb,
        "_load_summary_index",
        lambda: [
            {"path": "display_types.md", "summary": "LCD TFT PMOLED 等显示技术"},
            {"path": "voice_notes.md", "summary": "周会语音转写"},
        ],
    )

    async def fake_vector_scores(entries, query):
        return {"display_types.md": 0.86, "voice_notes.md": 0.04}

    monkeypatch.setattr(kb, "_compute_vector_scores", fake_vector_scores)

    result = asyncio.run(kb.search_paths("self emissive panel alternatives", top_k=2))
    payload = json.loads(result)

    assert payload["candidates"][0]["path"] == "display_types.md"
    assert payload["candidates"][0]["match_mode"] == "vector"
    assert payload["candidates"][0]["vector_score"] == 0.86


def test_update_summary_entry_invalidates_embedding_cache():
    temp_dir = _make_local_tmp_dir("invalidate_embedding_cache")
    kb = KnowledgeBase(base_path=str(temp_dir))
    kb.summary_json_file = temp_dir / "summary_demo.json"
    kb._embedding_cache_state = {"provider": "ragas_embedding", "model": "bge-m3", "entries": {}}

    kb.update_summary_entry("Image-Notes/demo.md", "image summary")

    assert kb._embedding_cache_state is None
    assert kb.summary_json_file.exists()


def test_extract_terms_filters_common_english_stopwords():
    kb = KnowledgeBase(base_path="tests/_tmp/stopwords")

    terms = kb._extract_terms("the reliability and compatibility of tool invocation")

    assert "the" not in terms
    assert "and" not in terms
    assert "of" not in terms
    assert "reliability" in terms
    assert "compatibility" in terms


def test_rank_sections_can_use_vector_signal_to_promote_better_evidence():
    kb = KnowledgeBase(base_path="tests/_tmp/section_rank")
    sections = [
        ("intro", "This section mentions AMOLED briefly."),
        ("details", "This section explains self emissive display technologies in depth."),
    ]
    query_terms = kb._extract_terms("self emissive display alternatives")
    query_embedding = [1.0, 0.0]
    section_vectors = [
        [0.2, 1.0],
        [0.95, 0.05],
    ]

    ranked = kb._rank_sections(
        sections,
        query_terms,
        query_embedding=query_embedding,
        section_vectors=section_vectors,
    )

    assert ranked[0][1] == "details"
