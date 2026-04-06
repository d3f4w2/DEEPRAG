export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatBudgetConfig {
  max_total_tokens?: number;
  max_latency_ms?: number;
  price_per_1m_tokens?: number;
  cost_multiplier?: number;
}

export interface ChatRequest {
  messages: Message[];
  provider?: string;
  model?: string;
  budget?: ChatBudgetConfig;
}

export interface VoiceDraftRequest {
  transcript: string;
  author?: string;
  provider?: string;
}

export interface VoiceDraftResponse {
  polished_text: string;
  summary: string;
  warning?: string;
}

export interface VoiceIngestRequest {
  transcript: string;
  summary: string;
  author: string;
  source: string;
  occurred_at?: string;
  raw_transcript?: string;
}

export interface VoiceIngestResponse {
  status: string;
  file_path: string;
  created_at: string;
}

export interface ImageDraftResponse {
  ocr_text: string;
  visual_summary: string;
  visual_description: string;
  tags: string[];
  retrieval_keywords: string[];
  ocr_line_count: number;
  image_width: number;
  image_height: number;
}

export interface ImageIngestResponse {
  status: string;
  file_path: string;
  image_path: string;
  created_at: string;
}

export interface Provider {
  id: string;
  name: string;
  models: string[];
}

export interface KnowledgeBaseInfo {
  summary: string;
  file_tree: FileNode;
}

export interface FileNode {
  type: 'file' | 'directory';
  name: string;
  path?: string;
  children?: FileNode[];
}

export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface StreamChunk {
  type:
    | 'content'
    | 'tool_calls'
    | 'tool_results'
    | 'reasoning_stage'
    | 'usage'
    | 'retrieval_critic'
    | 'budget_update'
    | 'budget_guard'
    | 'budget_summary'
    | 'done';
  content?: string;
  tool_calls?: ToolCall[];
  results?: ToolResult[];
  usage?: Record<string, any>;
  decision?: 'accept' | 'revise' | 'refuse';
  mode?: 'llm' | 'fallback' | 'rule' | string;
  stop?: boolean;
  reason?: string;
  confidence?: number;
  evidence_items?: number;
  distinct_files?: number;
  matched_evidence_items?: number;
  total_snippet_chars?: number;
  uncertain_answer?: boolean;
  evidence_sufficient?: boolean;
  rule_reason?: string;
  query_term_hits?: number;
  query_terms?: number;
  retrieval_rounds?: number;
  max_retrieval_rounds?: number;
  tool_call_rounds?: number;
  max_tool_call_rounds?: number;
  retry_count_total?: number;
  retry_exhausted_count?: number;
  cache_hit_count?: number;
  auto_retrieval_rounds?: number;
  max_auto_retrieval_rounds?: number;
  stage_key?: string;
  stage_label?: string;
  title?: string;
  summary?: string;
  status?: string;
  badge?: string;
  order?: number;
  updated_at?: string;
  metrics?: ReasoningStageMetric[];
  triggered?: boolean;
  limits?: {
    max_total_tokens?: number;
    max_latency_ms?: number;
  };
  pricing?: {
    price_per_1m_tokens?: number;
    cost_multiplier?: number;
  };
  cost_estimate_usd?: number;
}

export interface ToolResult {
  role: string;
  tool_call_id: string;
  content: string;
}

export interface RetrievalCritic {
  decision: 'accept' | 'revise' | 'refuse';
  stop: boolean;
  mode: string;
  reason: string;
  confidence: number;
  evidence_items: number;
  distinct_files: number;
  matched_evidence_items?: number;
  total_snippet_chars?: number;
  uncertain_answer?: boolean;
  evidence_sufficient?: boolean;
  rule_reason?: string;
  query_term_hits: number;
  query_terms: number;
  retrieval_rounds?: number;
  max_retrieval_rounds?: number;
  tool_call_rounds?: number;
  max_tool_call_rounds?: number;
  retry_count_total?: number;
  retry_exhausted_count?: number;
  cache_hit_count?: number;
  auto_retrieval_rounds?: number;
  max_auto_retrieval_rounds?: number;
}

export interface ReasoningStageMetric {
  label: string;
  value: string;
  tone?: 'neutral' | 'good' | 'warn' | 'risk' | string;
}

export interface ReasoningStage {
  stage_key: string;
  stage_label: string;
  title: string;
  summary: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | string;
  mode: string;
  badge: string;
  order: number;
  updated_at: string;
  metrics: ReasoningStageMetric[];
}

export interface BudgetSummary {
  triggered: boolean;
  reason: string;
  total_tokens: number;
  elapsed_ms: number;
  max_total_tokens: number;
  max_latency_ms: number;
  price_per_1m_tokens: number;
  cost_multiplier: number;
  cost_estimate_usd: number;
}

export interface EvaluationDataset {
  name: string;
  path: string;
  question_count: number;
  dataset_type?: string;
  updated_at: string;
}

export interface EvaluationSummaryItem {
  run_name: string;
  summary_path: string;
  report_path: string;
  analysis_path: string;
  created_at: string;
  accuracy: number;
  evidence_hit_rate: number;
  faithfulness?: number;
  context_recall?: number;
  answer_correctness?: number;
  avg_token: number;
  avg_latency_ms: number;
}

export interface EvaluationStartRequest {
  dataset_path: string;
  run_name?: string;
  provider?: string;
  compare_summary_path?: string;
  max_questions?: number;
  timeout_sec?: number;
  generate_analysis?: boolean;
  base_url?: string;
}

export interface EvaluationFailureItem {
  id: string;
  问题?: string;
  错误?: string;
  忠诚度?: number | null;
  上下文召回率?: number | null;
  答案准确度?: number | null;
  期望证据文件?: string[];
  引用证据文件?: string[];
  检索证据文件?: string[];
  答案预览?: string;
  [key: string]: unknown;
}

export interface EvaluationJob {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  progress_percent: number;
  current_index: number;
  current_total: number;
  current_question: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  run_name: string;
  dataset_path: string;
  provider?: string;
  compare_summary_path?: string;
  timeout_sec?: number;
  max_questions?: number;
  generate_analysis?: boolean;
  base_url?: string;
  summary?: Record<string, unknown>;
  report_markdown?: string;
  analysis_markdown?: string;
  outputs?: Record<string, string>;
  failures?: EvaluationFailureItem[];
  recent_logs?: string[];
}
