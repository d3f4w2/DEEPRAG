from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.config import settings
from backend.llm_provider import LLMProvider


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class EvaluationManager:
    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        self.eval_dir = self.root_dir / "评测"
        self.results_dir = self.eval_dir / "结果"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._max_jobs = 50
        self._eval_module = self._load_eval_module()

        self.min_faithfulness = float((settings.__dict__.get("eval_min_faithfulness") or 0.75))
        self.min_context_recall = float((settings.__dict__.get("eval_min_context_recall") or 0.75))
        self.min_answer_correctness = float((settings.__dict__.get("eval_min_answer_correctness") or 0.75))

    def _load_eval_module(self) -> Any:
        module_path = self.eval_dir / "运行小评测.py"
        if not module_path.exists():
            raise FileNotFoundError(f"评测脚本不存在: {module_path}")

        spec = importlib.util.spec_from_file_location("evaluation_script", str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载评测脚本: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _to_abs_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        return path

    def _to_rel_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root_dir)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def _safe_run_name(self, run_name: str) -> str:
        candidate = (run_name or "").strip()
        if not candidate:
            candidate = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", candidate, flags=re.UNICODE).strip("_")
        return safe or datetime.now().strftime("%Y%m%d_%H%M%S")

    def _ensure_unique_run_name(self, run_name: str) -> str:
        base = self._safe_run_name(run_name)
        if not any((self.results_dir / f"{base}_{suffix}.json").exists() for suffix in ("汇总", "明细")):
            return base

        idx = 2
        while True:
            candidate = f"{base}_{idx}"
            if not any((self.results_dir / f"{candidate}_{suffix}.json").exists() for suffix in ("汇总", "明细")):
                return candidate
            idx += 1

    def _trim_jobs_if_needed(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        sorted_ids = sorted(
            self._jobs.keys(),
            key=lambda k: self._jobs[k].get("created_at", ""),
        )
        for old_id in sorted_ids[: max(0, len(sorted_ids) - self._max_jobs)]:
            self._jobs.pop(old_id, None)

    def _get_job_copy(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return json.loads(json.dumps(job, ensure_ascii=False))

    def _update_job(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                job[key] = value

    def _append_job_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            logs = job.setdefault("recent_logs", [])
            logs.append(line)
            if len(logs) > 120:
                del logs[: len(logs) - 120]

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []
        if not self.eval_dir.exists():
            return datasets

        for file_path in sorted(self.eval_dir.glob("*.json")):
            if any(marker in file_path.name for marker in ("_汇总", "_明细")):
                continue

            question_count = 0
            dataset_type = "unknown"
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                questions = payload.get("问题列表", []) if isinstance(payload, dict) else payload
                if isinstance(questions, list):
                    question_count = len(questions)
                    if questions:
                        first = questions[0] if isinstance(questions[0], dict) else {}
                        if isinstance(first, dict):
                            if "参考答案" in first or "reference" in first:
                                dataset_type = "ragas"
                            elif "判分关键词组" in first:
                                dataset_type = "legacy"
            except Exception:
                question_count = 0
                dataset_type = "invalid"

            datasets.append(
                {
                    "name": file_path.name,
                    "path": self._to_rel_path(file_path),
                    "question_count": question_count,
                    "dataset_type": dataset_type,
                    "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
        return datasets

    def list_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        if not self.results_dir.exists():
            return summaries

        for summary_path in sorted(
            self.results_dir.glob("*_汇总.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            run_name = str(summary.get("run_name") or summary_path.name.replace("_汇总.json", ""))
            report_path = self.results_dir / f"{run_name}_报告.md"
            analysis_path = self.results_dir / f"{run_name}_复盘.md"
            metrics = summary.get("指标", {})

            faithfulness = _safe_float(metrics.get("忠诚度", metrics.get("准确率", 0.0)))
            context_recall = _safe_float(metrics.get("上下文召回率", metrics.get("证据命中率", 0.0)))
            answer_correctness = _safe_float(metrics.get("答案准确度", metrics.get("准确率", 0.0)))
            avg_token = _safe_float(metrics.get("平均token", 0.0))
            avg_latency_ms = _safe_float(metrics.get("平均延迟_ms", 0.0))

            summaries.append(
                {
                    "run_name": run_name,
                    "summary_path": self._to_rel_path(summary_path),
                    "report_path": self._to_rel_path(report_path) if report_path.exists() else "",
                    "analysis_path": self._to_rel_path(analysis_path) if analysis_path.exists() else "",
                    "created_at": summary.get("created_at", ""),
                    # 兼容老前端字段
                    "accuracy": answer_correctness,
                    "evidence_hit_rate": context_recall,
                    # 新字段
                    "faithfulness": faithfulness,
                    "context_recall": context_recall,
                    "answer_correctness": answer_correctness,
                    "avg_token": avg_token,
                    "avg_latency_ms": avg_latency_ms,
                }
            )
        return summaries

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
            return json.loads(json.dumps(jobs, ensure_ascii=False))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._get_job_copy(job_id)

    def start_job(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        dataset_path = str(request_payload.get("dataset_path") or "").strip()
        if not dataset_path:
            raise ValueError("dataset_path 不能为空")

        run_name = self._ensure_unique_run_name(str(request_payload.get("run_name") or ""))
        provider = str(request_payload.get("provider") or "").strip() or None
        compare_summary_path = str(request_payload.get("compare_summary_path") or "").strip()
        timeout_sec = float(request_payload.get("timeout_sec") or 180.0)
        max_questions = int(request_payload.get("max_questions") or 0)
        generate_analysis = bool(request_payload.get("generate_analysis", True))
        base_url = str(request_payload.get("base_url") or "").strip() or "http://127.0.0.1:8000"

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress_percent": 0.0,
            "current_index": 0,
            "current_total": 0,
            "current_question": "",
            "created_at": _now_iso(),
            "started_at": "",
            "finished_at": "",
            "error": "",
            "run_name": run_name,
            "dataset_path": dataset_path,
            "provider": provider or "",
            "compare_summary_path": compare_summary_path,
            "timeout_sec": timeout_sec,
            "max_questions": max_questions,
            "generate_analysis": generate_analysis,
            "base_url": base_url,
            "summary": {},
            "report_markdown": "",
            "analysis_markdown": "",
            "outputs": {},
            "failures": [],
            "recent_logs": [],
        }

        with self._lock:
            self._jobs[job_id] = job
            self._trim_jobs_if_needed()

        worker = threading.Thread(
            target=self._run_job_thread,
            args=(job_id,),
            daemon=True,
        )
        worker.start()
        return self.get_job(job_id) or {}

    def _extract_failures(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for item in results:
            error = str(item.get("错误") or "")
            faith = to_float_local(item.get("忠诚度"))
            recall = to_float_local(item.get("上下文召回率"))
            correct = to_float_local(item.get("答案准确度"))

            failed = bool(error)
            if faith is None or recall is None or correct is None:
                failed = True
            if faith is not None and faith < self.min_faithfulness:
                failed = True
            if recall is not None and recall < self.min_context_recall:
                failed = True
            if correct is not None and correct < self.min_answer_correctness:
                failed = True

            if not failed:
                continue

            failures.append(
                {
                    "id": item.get("id", ""),
                    "问题": item.get("问题", ""),
                    "错误": error,
                    "忠诚度": faith,
                    "上下文召回率": recall,
                    "答案准确度": correct,
                    "期望证据文件": item.get("期望证据文件", []),
                    "引用证据文件": item.get("引用证据文件", []),
                    "检索证据文件": item.get("检索证据文件", []),
                    "答案预览": str(item.get("答案", ""))[:400],
                }
            )
        return failures

    async def _llm_text(self, provider_name: Optional[str], messages: List[Dict[str, str]]) -> str:
        provider = LLMProvider(provider=provider_name or settings.api_provider)
        text = ""
        async for chunk_str in provider.chat_completion(messages=messages, tools=None, stream=False):
            try:
                chunk = json.loads(chunk_str)
            except Exception:
                continue
            if chunk.get("type") == "content":
                text += chunk.get("content", "")
        return text.strip()

    def _generate_llm_analysis(
        self,
        provider_name: Optional[str],
        summary: Dict[str, Any],
        failures: List[Dict[str, Any]],
    ) -> str:
        prompt_payload = {
            "summary": summary,
            "failure_count": len(failures),
            "failures": failures[:12],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是资深 RAG 评测分析助手。请基于输入的评测汇总和失败题，输出简洁、可执行的复盘报告。"
                    "格式固定为 Markdown，包含：总体结论、主要问题（按影响排序）、优化动作（P0/P1/P2）、下一轮评测建议。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False),
            },
        ]
        try:
            return asyncio.run(self._llm_text(provider_name, messages))
        except Exception as exc:
            return f"### 复盘生成失败\n\n`{exc}`"

    def _run_job_thread(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        try:
            self._update_job(
                job_id,
                status="running",
                stage="evaluating",
                started_at=_now_iso(),
                progress_percent=2.0,
            )

            dataset_path = self._to_abs_path(str(job["dataset_path"]))
            if not dataset_path.exists():
                raise FileNotFoundError(f"评测集不存在: {dataset_path}")

            compare_summary = None
            compare_summary_path = str(job.get("compare_summary_path") or "").strip()
            if compare_summary_path:
                compare_abs = self._to_abs_path(compare_summary_path)
                if compare_abs.exists():
                    compare_summary = self._eval_module.load_summary(compare_abs)

            questions = self._eval_module.load_dataset(dataset_path)
            max_questions = int(job.get("max_questions") or 0)
            if max_questions > 0:
                questions = questions[:max_questions]
            if not questions:
                raise ValueError("评测集为空，无法运行。")

            total_questions = len(questions)
            self._update_job(job_id, current_index=0, current_total=total_questions, current_question="准备评测")

            def progress_callback(info: Dict[str, Any]) -> None:
                current = int(info.get("current") or 0)
                total = int(info.get("total") or total_questions)
                question = str(info.get("question") or "")
                eval_percent = 5.0 + (current / max(total, 1)) * 78.0
                self._update_job(
                    job_id,
                    status="running",
                    stage="evaluating",
                    progress_percent=round(min(eval_percent, 83.0), 2),
                    current_index=current,
                    current_total=total,
                    current_question=question,
                )

            def log_callback(line: str) -> None:
                self._append_job_log(job_id, line)

            self._append_job_log(job_id, f"加载评测集: {self._to_rel_path(dataset_path)}")
            self._append_job_log(job_id, f"题目数量: {total_questions}")

            results = self._eval_module.run_eval(
                questions=questions,
                base_url=str(job["base_url"]),
                provider=(job.get("provider") or None),
                timeout_sec=float(job.get("timeout_sec") or 180.0),
                progress_callback=progress_callback,
                log_callback=log_callback,
            )

            run_name = str(job["run_name"])
            summary = self._eval_module.summarize_results(
                results=results,
                run_name=run_name,
                base_url=str(job["base_url"]),
                provider=(job.get("provider") or None),
                dataset_path=dataset_path,
            )

            self._update_job(
                job_id,
                progress_percent=84.0,
                current_question="正在写入评测结果文件",
                summary=summary,
            )

            detail_payload = {
                "run_name": run_name,
                "created_at": summary.get("created_at", _now_iso()),
                "题目总数": len(results),
                "results": results,
            }

            detail_path = self.results_dir / f"{run_name}_明细.json"
            summary_path = self.results_dir / f"{run_name}_汇总.json"
            report_path = self.results_dir / f"{run_name}_报告.md"
            analysis_path = self.results_dir / f"{run_name}_复盘.md"

            detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            report_content = self._eval_module.build_markdown_report(
                summary=summary,
                detail_path=detail_path,
                summary_path=summary_path,
                compare_summary=compare_summary,
            )

            failures = self._extract_failures(results)
            analysis_markdown = ""
            if bool(job.get("generate_analysis", True)):
                self._update_job(
                    job_id,
                    status="running",
                    stage="analyzing",
                    progress_percent=90.0,
                    current_question="正在生成 LLM 复盘报告",
                )
                analysis_markdown = self._generate_llm_analysis(
                    provider_name=(job.get("provider") or None),
                    summary=summary,
                    failures=failures,
                ).strip()
                analysis_path.write_text(analysis_markdown, encoding="utf-8")
                report_content = report_content + "\n\n---\n\n## LLM 复盘报告\n\n" + analysis_markdown

            report_path.write_text(report_content, encoding="utf-8")

            outputs = {
                "summary_path": self._to_rel_path(summary_path),
                "detail_path": self._to_rel_path(detail_path),
                "report_path": self._to_rel_path(report_path),
                "analysis_path": self._to_rel_path(analysis_path) if analysis_path.exists() else "",
            }

            self._append_job_log(job_id, "评测任务完成。")
            self._update_job(
                job_id,
                status="completed",
                stage="completed",
                progress_percent=100.0,
                finished_at=_now_iso(),
                current_index=total_questions,
                current_total=total_questions,
                current_question="已完成",
                outputs=outputs,
                report_markdown=report_content,
                analysis_markdown=analysis_markdown,
                failures=failures,
            )
        except Exception as exc:
            error_text = f"{exc}\n\n{traceback.format_exc()}"
            self._append_job_log(job_id, f"[ERROR] {exc}")
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                progress_percent=100.0,
                finished_at=_now_iso(),
                error=error_text,
                current_question="任务失败",
            )


def to_float_local(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if math_is_nan_or_inf(v):
        return None
    return v


def math_is_nan_or_inf(v: float) -> bool:
    return v != v or v in (float("inf"), float("-inf"))


evaluation_manager = EvaluationManager()
