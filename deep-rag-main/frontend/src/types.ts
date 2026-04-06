export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  messages: Message[];
  provider?: string;
  model?: string;
  // Note: temperature, max_tokens, stream have been removed
  // temperature and max_tokens can only be set through global configuration in backend .env file
  // stream is forced to true and cannot be modified
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
  type: 'content' | 'tool_calls' | 'tool_results' | 'usage' | 'retrieval_judge' | 'done';
  content?: string;
  tool_calls?: ToolCall[];
  results?: ToolResult[];
  usage?: Record<string, any>;
  stop?: boolean;
  reason?: string;
}

export interface ToolResult {
  role: string;
  tool_call_id: string;
  content: string;
}

export interface RetrievalJudge {
  stop: boolean;
  reason: string;
}

export interface EvaluationDataset {
  name: string;
  path: string;
  question_count: number;
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
  问题: string;
  准确: boolean;
  证据命中: boolean;
  错误: string;
  期望证据文件: string[];
  引用证据文件: string[];
  答案预览: string;
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
