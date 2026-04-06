from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings


ENTITY_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
ENTITY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "into",
    "their",
    "there",
    "about",
    "across",
    "through",
    "during",
    "while",
    "where",
    "which",
    "within",
    "region",
    "market",
    "annual",
    "revenue",
    "stores",
    "store",
    "province",
    "company",
    "product",
}


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _load_docs(kb_root: Path, max_files: int, max_chars: int, min_chars: int) -> list[Document]:
    docs: list[Document] = []
    markdown_files = sorted(kb_root.rglob("*.md"))[:max_files]
    for file_path in markdown_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(text.strip()) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        rel_path = str(file_path.relative_to(kb_root)).replace("\\", "/")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": rel_path},
            )
        )
    return docs


def _extract_entities(text: str, top_k: int = 36) -> list[str]:
    counts: dict[str, int] = {}
    for token in ENTITY_TOKEN_PATTERN.findall(text or ""):
        normalized = token.strip("-").lower()
        if len(normalized) < 3 or normalized in ENTITY_STOPWORDS:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:top_k]]


def _build_heuristic_knowledge_graph(docs: list[Document]) -> KnowledgeGraph:
    nodes: list[Node] = []
    for doc in docs:
        text = str(doc.page_content or "").strip()
        entities = _extract_entities(text)
        if not entities:
            source_text = str((doc.metadata or {}).get("source") or "document").replace("/", " ")
            entities = _extract_entities(source_text) or ["document"]

        nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": text,
                    "document_metadata": doc.metadata or {},
                    "entities": entities,
                },
            )
        )

    relationships: list[Relationship] = []
    for left_idx, left_node in enumerate(nodes):
        left_entities = set(left_node.properties.get("entities") or [])
        if not left_entities:
            continue

        for right_idx in range(left_idx + 1, len(nodes)):
            right_node = nodes[right_idx]
            right_entities = set(right_node.properties.get("entities") or [])
            if not right_entities:
                continue

            overlap = sorted(left_entities & right_entities)
            if not overlap:
                continue

            score = len(overlap) / max(1, min(len(left_entities), len(right_entities)))
            relationships.append(
                Relationship(
                    source=left_node,
                    target=right_node,
                    type="entities_overlap",
                    properties={
                        "entities_overlap_score": score,
                        "overlapped_items": [(item, item) for item in overlap[:20]],
                    },
                )
            )

    return KnowledgeGraph(nodes=nodes, relationships=relationships)


def _generate_without_embeddings(
    generator: TestsetGenerator,
    docs: list[Document],
    testset_size: int,
    run_config: RunConfig,
):
    kg = _build_heuristic_knowledge_graph(docs)
    generator.knowledge_graph = kg

    single_hop = SingleHopSpecificQuerySynthesizer(llm=generator.llm)
    multi_hop = MultiHopSpecificQuerySynthesizer(llm=generator.llm)

    distribution: list[tuple[Any, float]] = []
    try:
        if single_hop.get_node_clusters(kg):
            distribution.append((single_hop, 0.7))
    except Exception:
        pass
    try:
        if multi_hop.get_node_clusters(kg):
            distribution.append((multi_hop, 0.3))
    except Exception:
        pass

    if not distribution:
        raise ValueError("无法构建可用的 query distribution（entities/overlap 不足）")

    total = sum(weight for _, weight in distribution)
    normalized_distribution = [(synth, weight / total) for synth, weight in distribution]

    if not generator.persona_list:
        generator.persona_list = [
            Persona(name="产品经理", role_description="关注功能能力、参数差异与适用场景"),
            Persona(name="采购经理", role_description="关注价格、供应、合同与交付风险"),
            Persona(name="技术支持工程师", role_description="关注技术细节、兼容性与故障排查"),
        ]

    return generator.generate(
        testset_size=testset_size,
        query_distribution=normalized_distribution,
        run_config=run_config,
        raise_exceptions=False,
    )


def _embedding_endpoint_available(embeddings: OpenAIEmbeddings) -> bool:
    try:
        embeddings.embed_query("healthcheck")
        return True
    except Exception:
        return False


def _build_payload(samples: list[dict[str, Any]], description: str) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for idx, item in enumerate(samples, start=1):
        question = str(item.get("user_input") or item.get("问题") or "").strip()
        reference = str(item.get("reference") or item.get("参考答案") or "").strip()
        if not question or not reference:
            continue

        reference_contexts = _coerce_str_list(item.get("reference_contexts") or item.get("参考上下文"))
        expected_files = _coerce_str_list(item.get("expected_files") or item.get("期望证据文件"))
        synth = str(item.get("synthesizer_name") or item.get("来源类型") or "").strip()
        query_style = str(item.get("query_style") or "").strip()
        query_length = str(item.get("query_length") or "").strip()

        questions.append(
            {
                "id": f"R{idx:03d}",
                "问题": question,
                "参考答案": reference,
                "参考上下文": reference_contexts,
                "期望证据文件": expected_files,
                "来源类型": synth,
                "query_style": query_style,
                "query_length": query_length,
            }
        )

    return {
        "版本": "ragas-v1",
        "说明": description,
        "问题列表": questions,
    }


def build_via_ragas_generator(
    output_path: Path,
    kb_root: Path,
    testset_size: int,
    provider: str | None,
    embedding_model: str | None,
    max_files: int,
    max_chars: int,
    min_chars: int,
    json_mode: bool,
    allow_fallback: bool,
    max_workers: int,
    max_retries: int,
    run_timeout: int,
) -> dict[str, Any]:
    provider_name = (provider or settings.api_provider).strip()
    cfg = settings.get_provider_config(provider_name)
    api_key = str(cfg.get("api_key") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip() or None
    model = str(cfg.get("model") or "").strip()
    emb_model = (embedding_model or "").strip() or f"{provider_name.upper()}_EMBEDDING_MODEL"
    if emb_model == f"{provider_name.upper()}_EMBEDDING_MODEL":
        emb_model = os.getenv(emb_model) or os.getenv("RAGAS_EMBEDDING_MODEL") or "text-embedding-3-small"
    emb_api_key = (
        os.getenv(f"{provider_name.upper()}_EMBEDDING_API_KEY")
        or os.getenv("RAGAS_EMBEDDING_API_KEY")
        or api_key
    )
    emb_base_url = (
        os.getenv(f"{provider_name.upper()}_EMBEDDING_BASE_URL")
        or os.getenv("RAGAS_EMBEDDING_BASE_URL")
        or base_url
    )

    if not api_key:
        raise ValueError(f"缺少 {provider_name.upper()}_API_KEY")
    if not model:
        raise ValueError(f"缺少 {provider_name.upper()}_MODEL")
    if not emb_api_key:
        raise ValueError("缺少 Embedding API Key（可设置 RAGAS_EMBEDDING_API_KEY）")

    docs = _load_docs(kb_root=kb_root, max_files=max_files, max_chars=max_chars, min_chars=min_chars)
    if not docs:
        raise ValueError(f"知识库文档为空或不满足最小长度，路径: {kb_root}")

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0.0,
    }
    if json_mode:
        # RAGAS transforms rely on structured outputs; force JSON mode for better stability.
        llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    llm = ChatOpenAI(**llm_kwargs)
    embeddings = OpenAIEmbeddings(model=emb_model, api_key=emb_api_key, base_url=emb_base_url)
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    run_config = RunConfig(
        max_workers=max(1, int(max_workers)),
        max_retries=max(1, int(max_retries)),
        timeout=max(30, int(run_timeout)),
    )

    if not _embedding_endpoint_available(embeddings):
        if not allow_fallback:
            raise ValueError(
                "embeddings endpoint 不可用。请检查 `RAGAS_EMBEDDING_MODEL / "
                "RAGAS_EMBEDDING_BASE_URL / RAGAS_EMBEDDING_API_KEY` 配置。"
            )
        print("[warn] embeddings endpoint 不可用，按参数启用无 embeddings 兜底路径。")
        testset = _generate_without_embeddings(
            generator=generator,
            docs=docs,
            testset_size=testset_size,
            run_config=run_config,
        )
    else:
        try:
            testset = generator.generate_with_langchain_docs(
                documents=docs,
                testset_size=testset_size,
                run_config=run_config,
                raise_exceptions=False,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            print(f"[warn] 标准 RAGAS transform 失败，按参数切换兜底路径：{type(exc).__name__}: {exc}")
            testset = _generate_without_embeddings(
                generator=generator,
                docs=docs,
                testset_size=testset_size,
                run_config=run_config,
            )
    records = testset.to_list()

    payload = _build_payload(
        samples=records,
        description=(
            "由 RAGAS testset generator 自动生成。"
            "字段已转换为 Deep RAG RAGAS 评测格式（问题/参考答案/参考上下文）。"
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def convert_legacy_dataset(legacy_path: Path, output_path: Path) -> dict[str, Any]:
    if not legacy_path.exists():
        raise FileNotFoundError(f"legacy dataset 不存在: {legacy_path}")

    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    raw_questions = payload.get("问题列表", [])
    if not isinstance(raw_questions, list):
        raise ValueError("legacy dataset 格式错误：`问题列表` 不是 list")

    converted: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id") or f"R{idx:03d}")
        question = str(item.get("问题") or "").strip()
        keyword_groups = item.get("判分关键词组") or []
        expected_files = _coerce_str_list(item.get("期望证据文件"))
        if not question:
            continue

        key_tokens: list[str] = []
        for group in keyword_groups:
            if isinstance(group, list) and group:
                token = str(group[0]).strip()
                if token:
                    key_tokens.append(token)
            elif isinstance(group, str) and group.strip():
                key_tokens.append(group.strip())
        reference = "；".join(dict.fromkeys(key_tokens))
        if not reference:
            reference = "请根据知识库证据回答。"

        converted.append(
            {
                "id": qid,
                "问题": question,
                "参考答案": reference,
                "参考上下文": [],
                "期望证据文件": expected_files,
                "来源类型": "legacy-convert",
            }
        )

    out = {
        "版本": "ragas-v1",
        "说明": (
            "由 legacy 小评测集自动转换生成。"
            "注意：参考答案来自旧关键词，不代表高质量 gold answer，建议后续用 RAGAS 自动生成器重建。"
        ),
        "问题列表": converted,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="构建 RAGAS 评测集")
    parser.add_argument(
        "--mode",
        choices=["ragas", "legacy-convert"],
        default="ragas",
        help="ragas: 用 RAGAS testset generator 自动生成；legacy-convert: 从旧数据集转换",
    )
    parser.add_argument("--output", default=str(script_dir / "ragas评测集_40题.json"))
    parser.add_argument("--size", type=int, default=40, help="目标题目数，仅 ragas 模式生效")
    parser.add_argument("--provider", default="", help="用于生成题目的 provider")
    parser.add_argument("--embedding-model", default="", help="用于 testset generation 的 embedding model")
    parser.add_argument("--kb-root", default=str((Path(settings.knowledge_base_chunks)).resolve()))
    parser.add_argument("--max-files", type=int, default=120)
    parser.add_argument("--max-chars-per-file", type=int, default=1200)
    parser.add_argument("--min-chars-per-file", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=4, help="RAGAS 并发 worker 数")
    parser.add_argument("--max-retries", type=int, default=20, help="RAGAS 失败重试次数")
    parser.add_argument("--run-timeout", type=int, default=180, help="单任务超时（秒）")
    parser.add_argument(
        "--disable-json-mode",
        action="store_true",
        help="关闭 JSON response_format（默认开启以提升 RAGAS 生成稳定性）",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="当 embeddings 或 transform 链失败时，允许切换到无 embeddings 的兜底生成路径",
    )
    parser.add_argument("--legacy-dataset", default=str(script_dir / "小评测集_40题.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()

    if args.mode == "legacy-convert":
        payload = convert_legacy_dataset(
            legacy_path=Path(args.legacy_dataset).resolve(),
            output_path=output_path,
        )
        print(f"转换完成: {output_path}")
        print(f"题目数: {len(payload.get('问题列表', []))}")
        return

    payload = build_via_ragas_generator(
        output_path=output_path,
        kb_root=Path(args.kb_root).resolve(),
        testset_size=max(1, int(args.size)),
        provider=args.provider.strip() or None,
        embedding_model=args.embedding_model.strip() or None,
        max_files=max(1, int(args.max_files)),
        max_chars=max(300, int(args.max_chars_per_file)),
        min_chars=max(1, int(args.min_chars_per_file)),
        json_mode=not args.disable_json_mode,
        allow_fallback=bool(args.allow_fallback),
        max_workers=max(1, int(args.max_workers)),
        max_retries=max(1, int(args.max_retries)),
        run_timeout=max(30, int(args.run_timeout)),
    )
    print(f"生成完成: {output_path}")
    print(f"题目数: {len(payload.get('问题列表', []))}")


if __name__ == "__main__":
    main()
