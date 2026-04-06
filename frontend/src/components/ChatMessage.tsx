import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronRight, Wrench, FileText, Copy, Check, Sparkles, ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { BudgetSummary, Message, ReasoningStage, RetrievalCritic } from '../types';
import { API_BASE_URL } from '../api';
import './ChatMessage.css';

interface ChatMessageProps {
  message: Message;
  toolCalls?: any[];
  toolResults?: any[];
  reasoningStages?: ReasoningStage[];
  retrievalCritic?: RetrievalCritic;
  budgetSummary?: BudgetSummary;
}

interface EvidenceItem {
  filePath: string;
  snippet: string;
  imagePath?: string;
}

interface ParsedAssistantAnswer {
  answerText: string;
  evidenceItems: EvidenceItem[];
}

const EVIDENCE_HEADER_REGEX = /(?:^|\n)#{2,3}\s*(证据|evidence)\s*\n/i;
type StageStatus = 'pending' | 'running' | 'completed' | 'failed';
type JudgeDecision = 'accept' | 'revise' | 'refuse';
type JudgeTone = 'pass' | 'hold' | 'reject';

const buildKnowledgeFileUrl = (relativePath: string): string => {
  const normalized = (relativePath || '').trim().replace(/\\/g, '/').replace(/^\/+/, '');
  if (!normalized) {
    return '';
  }
  const encodedPath = normalized
    .split('/')
    .filter(Boolean)
    .map((seg) => encodeURIComponent(seg))
    .join('/');
  return `${API_BASE_URL}/knowledge-base/file/${encodedPath}`;
};

const parseAssistantAnswerWithEvidence = (content: string): ParsedAssistantAnswer => {
  if (!content?.trim()) {
    return { answerText: '', evidenceItems: [] };
  }

  const headerMatch = EVIDENCE_HEADER_REGEX.exec(content);
  if (!headerMatch || headerMatch.index === undefined) {
    return { answerText: content, evidenceItems: [] };
  }

  const answerText = content.slice(0, headerMatch.index).trim();
  const evidenceText = content.slice(headerMatch.index + headerMatch[0].length).trim();
  const evidenceBlocks = evidenceText
    .split(/\n(?=\d+\.\s*(来源文件|source file)\s*:)/i)
    .map((item) => item.trim())
    .filter(Boolean);

  const evidenceItems: EvidenceItem[] = [];
  evidenceBlocks.forEach((block) => {
    const fileMatch = block.match(/\d+\.\s*(来源文件|source file)\s*:\s*`?([^\n`]+)`?/i);
    if (!fileMatch) {
      return;
    }

    const imageMatch = block.match(/(?:原始图片|image)\s*:\s*`?([^\n`]+)`?/i);
    const snippetMatch = block.match(/(?:证据片段|snippet)\s*:\s*["“]?([\s\S]*?)["”]?\s*$/i);
    const filePath = fileMatch[2].trim();
    const imagePath = (imageMatch?.[1] || '').trim();
    let snippet = (snippetMatch?.[1] || '').trim();
    if (!snippet) {
      snippet = block
        .split('\n')
        .slice(1)
        .join(' ')
        .replace(/(?:证据片段|snippet)\s*:/i, '')
        .trim();
    }

    if (!filePath || !snippet) {
      return;
    }

    evidenceItems.push({
      filePath,
      snippet,
      imagePath: imagePath || undefined,
    });
  });

  return {
    answerText: answerText || content.trim(),
    evidenceItems,
  };
};

const normalizeStageStatus = (status: string): StageStatus => {
  const value = (status || '').trim().toLowerCase();
  if (value === 'pending' || value === 'running' || value === 'completed' || value === 'failed') {
    return value;
  }
  return 'running';
};

const stageStatusLabel = (status: StageStatus, stageKey = ''): string => {
  const normalizedKey = (stageKey || '').trim().toLowerCase();
  if (normalizedKey === 'critic' || normalizedKey === 'judge') {
    if (status === 'pending') return '待评审';
    if (status === 'running') return '评审中';
    if (status === 'completed') return '已放行';
    return '已拒绝';
  }
  if (status === 'pending') return '待处理';
  if (status === 'running') return '进行中';
  if (status === 'completed') return '已完成';
  return '失败';
};

const stageKeyLabel = (stageKey: string, fallback?: string): string => {
  const normalized = (stageKey || '').trim().toLowerCase();
  if (normalized === 'plan') return 'Plan';
  if (normalized === 'execute') return 'Execute';
  if (normalized === 'critic' || normalized === 'judge') return 'Judge';
  return fallback || normalized.toUpperCase() || 'Stage';
};

const normalizeJudgeDecision = (decision: unknown): JudgeDecision => {
  const normalized = String(decision || '').trim().toLowerCase();
  if (normalized === 'accept' || normalized === 'revise' || normalized === 'refuse') {
    return normalized;
  }
  return 'revise';
};

const stripJudgePrefix = (text: string): string => {
  const value = (text || '').trim();
  if (!value) return '';
  return value
    .replace(/^注：证据分用于衡量检索充分性，不等于答案正确率。\s*/i, '')
    .replace(/^judge 理由[:：]\s*/i, '')
    .trim();
};

const judgeOutcomeMeta = (decision: JudgeDecision): { verdict: string; tone: JudgeTone } => {
  if (decision === 'accept') {
    return { verdict: '放行', tone: 'pass' };
  }
  if (decision === 'refuse') {
    return { verdict: '拒绝回答', tone: 'reject' };
  }
  return { verdict: '暂不放行（继续检索）', tone: 'hold' };
};

const resolveJudgeOutcome = (
  stage: ReasoningStage,
  critic?: RetrievalCritic,
): { decision: JudgeDecision; verdict: string; tone: JudgeTone; reason: string } | null => {
  const stageKey = (stage.stage_key || '').trim().toLowerCase();
  if (stageKey !== 'critic' && stageKey !== 'judge') {
    return null;
  }

  let decision: JudgeDecision;
  if (critic) {
    decision = normalizeJudgeDecision(critic.decision || (critic.stop ? 'accept' : 'revise'));
  } else {
    const status = normalizeStageStatus(stage.status);
    decision = status === 'completed' ? 'accept' : status === 'failed' ? 'refuse' : 'revise';
  }

  const reason = stripJudgePrefix(
    String(critic?.rule_reason || critic?.reason || stage.summary || ''),
  );
  const { verdict, tone } = judgeOutcomeMeta(decision);
  return { decision, verdict, tone, reason };
};

const stageToneClass = (stageKey: string): string => {
  const normalized = (stageKey || '').trim().toLowerCase();
  if (normalized === 'plan') return 'plan';
  if (normalized === 'execute') return 'execute';
  if (normalized === 'critic') return 'critic';
  return 'generic';
};

const sortReasoningStages = (stages: ReasoningStage[]): ReasoningStage[] => {
  return [...stages].sort((a, b) => a.order - b.order || a.stage_key.localeCompare(b.stage_key));
};

const fallbackCriticStage = (critic?: RetrievalCritic): ReasoningStage[] => {
  if (!critic) {
    return [];
  }
  const decision = normalizeJudgeDecision(critic.decision || (critic.stop ? 'accept' : 'revise'));
  const fitScore = Math.max(0, Math.min(1, critic.confidence || 0));
  const fitPercent = Math.round(fitScore * 100);
  const fitLevel = fitScore >= 0.75 ? '高' : fitScore >= 0.55 ? '中' : '低';
  const fitTone: 'good' | 'warn' | 'risk' = fitScore >= 0.75 ? 'good' : fitScore >= 0.55 ? 'warn' : 'risk';
  const retrievalRounds = Math.max(0, Math.floor(Number(critic.retrieval_rounds || 0)));
  const maxRetrievalRounds = Math.max(0, Math.floor(Number(critic.max_retrieval_rounds || 0)));
  const toolCallRounds = Math.max(0, Math.floor(Number(critic.tool_call_rounds || 0)));
  const maxToolCallRounds = Math.max(0, Math.floor(Number(critic.max_tool_call_rounds || 0)));
  const retryCountTotal = Math.max(0, Math.floor(Number(critic.retry_count_total || 0)));
  const retryExhaustedCount = Math.max(0, Math.floor(Number(critic.retry_exhausted_count || 0)));
  const cacheHitCount = Math.max(0, Math.floor(Number(critic.cache_hit_count || 0)));
  const judgeMeta = judgeOutcomeMeta(decision);
  const judgeReason = stripJudgePrefix(
    String(critic.rule_reason || critic.reason || ''),
  ) || '未提供';
  const status: StageStatus =
    decision === 'accept'
      ? 'completed'
      : decision === 'refuse'
        ? 'failed'
        : 'running';
  const modeLabel = (critic.mode || 'llm').toUpperCase();
  const metrics = [
    {
      label: '检索轮次',
      value:
        maxRetrievalRounds > 0
          ? `${retrievalRounds}/${maxRetrievalRounds}`
          : `${retrievalRounds}`,
      tone: maxRetrievalRounds > 0 && retrievalRounds >= maxRetrievalRounds ? 'warn' : 'neutral',
    },
    {
      label: '工具轮次',
      value:
        maxToolCallRounds > 0
          ? `${toolCallRounds}/${maxToolCallRounds}`
          : `${toolCallRounds}`,
      tone: maxToolCallRounds > 0 && toolCallRounds >= maxToolCallRounds ? 'warn' : 'neutral',
    },
    {
      label: '失败重试',
      value: `${retryCountTotal} (耗尽 ${retryExhaustedCount})`,
      tone: retryExhaustedCount > 0 ? 'risk' : retryCountTotal > 0 ? 'warn' : 'neutral',
    },
    {
      label: '缓存命中',
      value: `${cacheHitCount}`,
      tone: cacheHitCount > 0 ? 'good' : 'neutral',
    },
    {
      label: 'Judge判定',
      value: judgeMeta.verdict,
      tone:
        decision === 'accept'
          ? 'good'
          : decision === 'refuse'
            ? 'risk'
            : 'warn',
    },
    {
      label: 'Evidence',
      value: `${critic.evidence_items || 0} / ${critic.distinct_files || 0}`,
      tone: 'neutral',
    },
    {
      label: 'Hit',
      value: `${critic.query_term_hits || 0}/${critic.query_terms || 0}`,
      tone: 'neutral',
    },
    {
      label: '证据等级',
      value: fitLevel,
      tone: fitTone,
    },
    {
      label: '匹配参考',
      value: `${fitPercent}% (非准确率)`,
      tone: 'neutral',
    },
  ];

  return [
    {
      stage_key: 'critic',
      stage_label: 'Judge',
      title:
        decision === 'accept'
          ? 'Judge 放行，允许回答'
          : decision === 'refuse'
            ? 'Judge 拒绝回答'
            : 'Judge 暂不放行，继续检索',
      summary: `Judge 理由：${judgeReason}`,
      status,
      mode: critic.mode || 'llm',
      badge: decision === 'accept' ? 'PASS' : decision === 'refuse' ? 'REJECT' : modeLabel,
      order: 30,
      updated_at: new Date().toISOString(),
      metrics,
    },
  ];
};

const getStageIcon = (stage: ReasoningStage, status: StageStatus) => {
  const key = (stage.stage_key || '').trim().toLowerCase();
  if (key === 'execute') {
    return Wrench;
  }
  if (key === 'critic' || key === 'judge') {
    if (status === 'completed') return ShieldCheck;
    if (status === 'failed') return ShieldX;
    return ShieldAlert;
  }
  return Sparkles;
};

const summarizeBudgetReason = (reason: string): string => {
  const text = (reason || '').trim();
  if (!text) return '';
  if (/token budget reached/i.test(text)) return 'Token 上限触发';
  if (/latency budget reached/i.test(text)) return '时延上限触发';
  const normalized = text.replace(/\s+/g, ' ');
  return normalized.length > 36 ? `${normalized.slice(0, 36)}...` : normalized;
};

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  toolCalls,
  toolResults,
  reasoningStages,
  retrievalCritic,
  budgetSummary,
}) => {
  const isUser = message.role === 'user';
  const [isToolCallExpanded, setIsToolCallExpanded] = useState(true);
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Check if it's a tool call message
  const isToolCallMessage = message.content?.startsWith('[TOOL]');
  const parsedAssistantAnswer = !isUser
    ? parseAssistantAnswerWithEvidence(message.content || '')
    : null;
  const evidenceItems = parsedAssistantAnswer?.evidenceItems || [];
  const hasEvidenceItems = evidenceItems.length > 0;
  const resolvedStages = reasoningStages && reasoningStages.length > 0
    ? sortReasoningStages(reasoningStages)
    : fallbackCriticStage(retrievalCritic);
  const onlyCriticStage =
    resolvedStages.length > 0 &&
    resolvedStages.every((stage) => {
      const key = (stage.stage_key || '').toLowerCase();
      return key === 'critic' || key === 'judge';
    });
  const pipelineTitle = onlyCriticStage ? 'Judge 评审' : 'Plan · Execute · Judge';
  const pipelineBadge = onlyCriticStage ? 'Judge Only' : 'LLM + Judge';
  const budgetReasonSummary = summarizeBudgetReason(budgetSummary?.reason || '');

  const handleCopy = async (content: string, index: number) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const toggleExpand = (index: number) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-icon">
        {isUser ? (
          <div className="icon-wrapper user-icon">
            <User size={18} />
          </div>
        ) : (
          <div className="icon-wrapper assistant-icon">
            <Bot size={18} />
          </div>
        )}
      </div>
      <div className="message-content">
        {isToolCallMessage && toolCalls ? (
          <div className="tool-call-container">
            <div
              className="tool-call-header"
              onClick={() => setIsToolCallExpanded(!isToolCallExpanded)}
            >
              <Wrench size={16} className="tool-icon" />
              <span className="tool-call-title">Tool Call</span>
              {isToolCallExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </div>
            {isToolCallExpanded && (
              <div className="tool-call-details">
                <div className="tool-section">
                  <div className="tool-section-title">
                    <Wrench size={14} />
                    <span>Input</span>
                  </div>
                  {toolCalls.map((toolCall, index) => {
                    try {
                      const args = JSON.parse(toolCall.function?.arguments || '{}');
                      const entries = Object.entries(args);
                      return (
                        <div key={index} className="tool-call-item">
                          <div className="tool-call-files">
                            {entries.length === 0 && (
                              <span className="file-badge">No arguments</span>
                            )}
                            {entries.map(([key, value]) => {
                              if (Array.isArray(value)) {
                                return value.map((item, i) => (
                                  <span key={`${key}-${i}`} className="file-badge">{`${key}: ${item}`}</span>
                                ));
                              }
                              return (
                                <span key={key} className="file-badge">
                                  {`${key}: ${String(value)}`}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      );
                    } catch (e) {
                      return null;
                    }
                  })}
                </div>

                {toolResults && toolResults.length > 0 && (
                  <div className="tool-section">
                    <div className="tool-section-title">
                      <FileText size={14} />
                      <span>Output</span>
                    </div>
                    {toolResults.map((result, index) => {
                      try {
                        const content = result.content || '';
                        const contentLength = content.length;
                        const MAX_DISPLAY_LENGTH = 500000;
                        const isTruncated = contentLength > MAX_DISPLAY_LENGTH;
                        const displayContent = isTruncated
                          ? content.substring(0, MAX_DISPLAY_LENGTH) + '\n\n... (content truncated for display)'
                          : content;
                        const isExpanded = expandedResults.has(index);

                        return (
                          <div key={index} className="tool-response">
                            <div className="tool-response-meta-row">
                              <span className="tool-response-meta">
                                {contentLength.toLocaleString()} characters retrieved
                                {isTruncated && ' (displaying first 500,000)'}
                              </span>
                              <div className="tool-response-actions">
                                <button
                                  className="icon-action-button"
                                  onClick={() => toggleExpand(index)}
                                  title={isExpanded ? 'Collapse' : 'Expand'}
                                >
                                  {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                </button>
                                <button
                                  className="icon-action-button copy-btn"
                                  onClick={() => handleCopy(content, index)}
                                  title="Copy full content to clipboard"
                                >
                                  {copiedIndex === index ? <Check size={16} /> : <Copy size={16} />}
                                </button>
                              </div>
                            </div>
                            {isExpanded && (
                              <div className="tool-response-content">
                                {displayContent}
                              </div>
                            )}
                          </div>
                        );
                      } catch (e) {
                        console.error('Error rendering tool result:', e);
                        return (
                          <div key={index} className="tool-response">
                            <div className="tool-response-meta" style={{ color: 'red' }}>
                              Error rendering content
                            </div>
                          </div>
                        );
                      }
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="message-text">
            {isUser ? (
              <div className="user-text">{message.content}</div>
            ) : (
              <div className="assistant-answer-container">
                {resolvedStages.length > 0 && (
                  <div className="reasoning-pipeline">
                    <div className="reasoning-pipeline-head">
                      <div className="reasoning-pipeline-title-wrap">
                        <Sparkles size={14} className="reasoning-pipeline-spark" />
                        <span className="reasoning-pipeline-main">{pipelineTitle}</span>
                      </div>
                      <span className="reasoning-pipeline-badge">{pipelineBadge}</span>
                    </div>
                    <div className="reasoning-stage-list">
                      {resolvedStages.map((stage, index) => {
                        const status = normalizeStageStatus(stage.status);
                        const Icon = getStageIcon(stage, status);
                        const stageTone = stageToneClass(stage.stage_key);
                        const visibleMetrics = stage.metrics.slice(0, 6);
                        const hiddenMetricCount = Math.max(0, stage.metrics.length - visibleMetrics.length);
                        const judgeOutcome = resolveJudgeOutcome(stage, retrievalCritic);

                        return (
                          <article
                            key={`${stage.stage_key}-${stage.updated_at}-${index}`}
                            className={`reasoning-stage-card ${stageTone} status-${status}`}
                          >
                            <div className="reasoning-stage-head">
                              <div className="reasoning-stage-kind">
                                <Icon size={13} />
                                <span>{stageKeyLabel(stage.stage_key, stage.stage_label)}</span>
                              </div>
                              <span className={`reasoning-stage-status status-${status}`}>
                                {stageStatusLabel(status, stage.stage_key)}
                              </span>
                            </div>
                            <div className="reasoning-stage-title">{stage.title}</div>
                            {stage.summary && (
                              <div className="reasoning-stage-summary" title={stage.summary}>
                                {stage.summary}
                              </div>
                            )}
                            {judgeOutcome && (
                              <div className={`judge-outcome judge-${judgeOutcome.tone}`}>
                                <div className="judge-outcome-title">
                                  Judge 判定: {judgeOutcome.verdict}
                                </div>
                                <div className="judge-outcome-reason" title={judgeOutcome.reason || '未提供'}>
                                  理由: {judgeOutcome.reason || '未提供'}
                                </div>
                              </div>
                            )}
                            <div className="reasoning-stage-metrics">
                              {stage.badge && (
                                <span className="reasoning-stage-chip soft">{stage.badge}</span>
                              )}
                              {visibleMetrics.map((metric, metricIndex) => (
                                <span
                                  key={`${stage.stage_key}-metric-${metricIndex}`}
                                  className={`reasoning-stage-chip tone-${metric.tone || 'neutral'}`}
                                >
                                  {metric.label}: {metric.value}
                                </span>
                              ))}
                              {hiddenMetricCount > 0 && (
                                <span className="reasoning-stage-chip more">+{hiddenMetricCount}</span>
                              )}
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                )}
                {budgetSummary && (
                  <div className={`budget-summary ${budgetSummary.triggered ? 'triggered' : 'normal'}`}>
                    <span className="budget-summary-main">
                      本次花费约 ${budgetSummary.cost_estimate_usd.toFixed(6)} · Token{' '}
                      {budgetSummary.total_tokens.toLocaleString()} · {budgetSummary.elapsed_ms}ms
                    </span>
                    <span className={`budget-status-badge ${budgetSummary.triggered ? 'triggered' : 'normal'}`}>
                      {budgetSummary.triggered ? 'Budget 触发' : 'Budget 正常'}
                    </span>
                    {budgetReasonSummary && (
                      <span className="budget-summary-reason" title={budgetSummary.reason}>
                        {budgetReasonSummary}
                      </span>
                    )}
                  </div>
                )}
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {hasEvidenceItems ? parsedAssistantAnswer?.answerText || '' : message.content || ''}
                  </ReactMarkdown>
                </div>

                {hasEvidenceItems ? (
                  <div className="evidence-panel">
                    <div className="evidence-panel-header">证据约束规范：来源文件 + 证据片段</div>
                    <div className="evidence-list">
                      {evidenceItems.map((item, index) => (
                        <div key={`${item.filePath}-${index}`} className="evidence-item">
                          <div className="evidence-source">
                            来源文件: <code>{item.filePath}</code>
                          </div>
                          {item.imagePath && (
                            <div className="evidence-image-wrap">
                              <div className="evidence-image-label">
                                原始图片: <code>{item.imagePath}</code>
                              </div>
                              <img
                                className="evidence-image"
                                src={buildKnowledgeFileUrl(item.imagePath)}
                                alt={`evidence-${index + 1}`}
                                loading="lazy"
                              />
                            </div>
                          )}
                          <div className="evidence-snippet">证据片段: "{item.snippet}"</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="evidence-missing">
                    证据约束规范：未检测到“来源文件 + 证据片段”结构
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
