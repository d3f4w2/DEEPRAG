import React, { useEffect, useMemo, useState } from 'react';
import { Play, RefreshCw, Loader2, CheckCircle2, AlertTriangle, Clock3 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { evaluationApi } from '../api';
import type {
  EvaluationDataset,
  EvaluationSummaryItem,
  EvaluationJob,
  EvaluationFailureItem,
} from '../types';
import './EvaluationPanel.css';

function toPercent(v: number | undefined): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '--';
  return `${(v * 100).toFixed(2)}%`;
}

function toNumber(v: number | undefined, digits = 2): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '--';
  return v.toFixed(digits);
}

function statusText(status: string): string {
  if (status === 'queued') return '排队中';
  if (status === 'running') return '运行中';
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  return status;
}

function stageText(stage: string): string {
  if (stage === 'queued') return '等待开始';
  if (stage === 'evaluating') return '执行评测';
  if (stage === 'analyzing') return 'LLM复盘';
  if (stage === 'completed') return '完成';
  if (stage === 'failed') return '失败';
  return stage || '--';
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (typeof error === 'string' && error) return error;
  if (error && typeof error === 'object') {
    const maybe = error as { message?: string; response?: { data?: { detail?: string } } };
    if (maybe.response?.data?.detail) return maybe.response.data.detail;
    if (maybe.message) return maybe.message;
  }
  return fallback;
}

function getMetric(metrics: Record<string, number>, key: string, fallback = 0): number {
  const value = metrics[key];
  if (typeof value === 'number' && !Number.isNaN(value)) return value;
  return fallback;
}

const EvaluationPanel: React.FC = () => {
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [summaries, setSummaries] = useState<EvaluationSummaryItem[]>([]);

  const [selectedDataset, setSelectedDataset] = useState('');
  const [runName, setRunName] = useState('');
  const [selectedCompareSummary, setSelectedCompareSummary] = useState('');
  const [provider, setProvider] = useState('');
  const [maxQuestions, setMaxQuestions] = useState<number>(0);
  const [timeoutSec, setTimeoutSec] = useState<number>(180);
  const [generateAnalysis, setGenerateAnalysis] = useState(true);
  const [baseUrl, setBaseUrl] = useState('');

  const [activeJobId, setActiveJobId] = useState('');
  const [activeJob, setActiveJob] = useState<EvaluationJob | null>(null);

  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const loadInitialData = async () => {
    setLoading(true);
    setError('');
    try {
      const [datasetRes, summaryRes, jobsRes] = await Promise.all([
        evaluationApi.listDatasets(),
        evaluationApi.listSummaries(),
        evaluationApi.listJobs(),
      ]);

      setDatasets(datasetRes.datasets || []);
      setSummaries(summaryRes.summaries || []);

      if (!selectedDataset && datasetRes.datasets?.length) {
        setSelectedDataset(datasetRes.datasets[0].path);
      }

      const runningJob = (jobsRes.jobs || []).find((j) => j.status === 'running' || j.status === 'queued');
      if (runningJob) {
        setActiveJobId(runningJob.job_id);
        setActiveJob(runningJob);
      } else if (!activeJob && (jobsRes.jobs || []).length > 0) {
        setActiveJob((jobsRes.jobs || [])[0]);
        setActiveJobId((jobsRes.jobs || [])[0].job_id);
      }
    } catch (e: unknown) {
      setError(getErrorMessage(e, '加载评测数据失败'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadInitialData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeJobRunning = useMemo(() => {
    if (!activeJob) return false;
    return activeJob.status === 'running' || activeJob.status === 'queued';
  }, [activeJob]);

  useEffect(() => {
    if (!activeJobId) return;

    let isMounted = true;
    let timer: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      try {
        const job = await evaluationApi.getJob(activeJobId);
        if (!isMounted) return;

        setActiveJob(job);

        if (job.status === 'completed' || job.status === 'failed') {
          if (timer) {
            clearInterval(timer);
            timer = null;
          }
          const summaryRes = await evaluationApi.listSummaries();
          if (!isMounted) return;
          setSummaries(summaryRes.summaries || []);
        }
      } catch {
        // keep polling silently
      }
    };

    void poll();
    timer = setInterval(() => {
      void poll();
    }, 2000);

    return () => {
      isMounted = false;
      if (timer) clearInterval(timer);
    };
  }, [activeJobId]);

  const handleStartEvaluation = async () => {
    if (!selectedDataset || starting) return;

    setStarting(true);
    setError('');
    try {
      const job = await evaluationApi.startJob({
        dataset_path: selectedDataset,
        run_name: runName.trim() || undefined,
        provider: provider.trim() || undefined,
        compare_summary_path: selectedCompareSummary || undefined,
        max_questions: maxQuestions > 0 ? maxQuestions : 0,
        timeout_sec: timeoutSec > 0 ? timeoutSec : 180,
        generate_analysis: generateAnalysis,
        base_url: baseUrl.trim() || undefined,
      });

      setActiveJobId(job.job_id);
      setActiveJob(job);
    } catch (e: unknown) {
      setError(getErrorMessage(e, '启动评测失败'));
    } finally {
      setStarting(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await loadInitialData();
    } finally {
      setRefreshing(false);
    }
  };

  const handleUseAsCompare = (summaryPath: string) => {
    setSelectedCompareSummary(summaryPath);
  };

  const metrics = ((activeJob?.summary as { 指标?: Record<string, number> } | undefined)?.指标) || {};
  const failures = (activeJob?.failures || []) as EvaluationFailureItem[];

  const faithfulness = getMetric(metrics, '忠诚度', getMetric(metrics, '准确率'));
  const contextRecall = getMetric(metrics, '上下文召回率', getMetric(metrics, '证据命中率'));
  const answerCorrectness = getMetric(metrics, '答案准确度', getMetric(metrics, '准确率'));
  const avgToken = getMetric(metrics, '平均token');
  const avgLatency = getMetric(metrics, '平均延迟_ms');

  return (
    <div className="evaluation-panel">
      <div className="evaluation-toolbar">
        <h2>评测工作台</h2>
        <button className="refresh-btn" onClick={handleRefresh} disabled={refreshing}>
          {refreshing ? <Loader2 className="spinner" size={16} /> : <RefreshCw size={16} />}
          刷新
        </button>
      </div>

      {error && <div className="evaluation-error">{error}</div>}

      <div className="evaluation-grid">
        <section className="panel-card">
          <h3>启动新评测</h3>
          {loading ? (
            <div className="panel-loading">
              <Loader2 className="spinner" size={18} />
              <span>加载中...</span>
            </div>
          ) : (
            <>
              <label>评测集</label>
              <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>
                {datasets.map((ds) => (
                  <option key={ds.path} value={ds.path}>
                    {ds.name} ({ds.question_count} 题{ds.dataset_type ? `, ${ds.dataset_type}` : ''})
                  </option>
                ))}
              </select>

              <label>运行名称（可选）</label>
              <input
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="例如 optimized_v2"
              />

              <label>对比基线（可选）</label>
              <select value={selectedCompareSummary} onChange={(e) => setSelectedCompareSummary(e.target.value)}>
                <option value="">不对比</option>
                {summaries.map((s) => (
                  <option key={s.summary_path} value={s.summary_path}>
                    {s.run_name} ({new Date(s.created_at || '').toLocaleString() || '--'})
                  </option>
                ))}
              </select>

              <div className="row">
                <div>
                  <label>Provider（可选）</label>
                  <input
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                    placeholder="留空=默认"
                  />
                </div>
                <div>
                  <label>题量上限</label>
                  <input
                    type="number"
                    min={0}
                    value={maxQuestions}
                    onChange={(e) => setMaxQuestions(Number(e.target.value || 0))}
                  />
                </div>
              </div>

              <div className="row">
                <div>
                  <label>单题超时(秒)</label>
                  <input
                    type="number"
                    min={30}
                    value={timeoutSec}
                    onChange={(e) => setTimeoutSec(Number(e.target.value || 180))}
                  />
                </div>
                <div>
                  <label>后端地址（可选）</label>
                  <input
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder="留空=后端默认"
                  />
                </div>
              </div>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={generateAnalysis}
                  onChange={(e) => setGenerateAnalysis(e.target.checked)}
                />
                <span>评测后自动生成 LLM 复盘报告</span>
              </label>

              <button
                className="start-btn"
                onClick={handleStartEvaluation}
                disabled={!selectedDataset || starting}
              >
                {starting ? <Loader2 className="spinner" size={16} /> : <Play size={16} />}
                一键评测
              </button>
            </>
          )}
        </section>

        <section className="panel-card">
          <h3>任务进度</h3>
          {!activeJob ? (
            <div className="empty-state">暂无任务，点击左侧“一键评测”开始。</div>
          ) : (
            <div className="job-status">
              <div className="job-head">
                <div>
                  <strong>{activeJob.run_name}</strong>
                  <div className="muted">{activeJob.dataset_path}</div>
                </div>
                <div className={`status-badge status-${activeJob.status}`}>
                  {activeJob.status === 'completed' && <CheckCircle2 size={14} />}
                  {activeJob.status === 'failed' && <AlertTriangle size={14} />}
                  {(activeJob.status === 'running' || activeJob.status === 'queued') && <Clock3 size={14} />}
                  <span>{statusText(activeJob.status)}</span>
                </div>
              </div>

              <div className="progress-wrap">
                <div className="progress-bar">
                  <div style={{ width: `${Math.max(0, Math.min(100, activeJob.progress_percent || 0))}%` }} />
                </div>
                <div className="muted">
                  {stageText(activeJob.stage)} · {toNumber(activeJob.progress_percent, 1)}%
                </div>
                <div className="muted">
                  {activeJob.current_index || 0}/{activeJob.current_total || 0} · {activeJob.current_question || '--'}
                </div>
              </div>

              {activeJob.error && <pre className="job-error">{activeJob.error}</pre>}

              <div className="logs-box">
                {(activeJob.recent_logs || []).slice(-10).map((line, i) => (
                  <div key={`${i}-${line}`} className="log-line">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      {activeJob && activeJob.status === 'completed' && (
        <>
          <section className="panel-card metrics-card">
            <h3>核心指标（RAGAS + 性能）</h3>
            <div className="metrics-grid">
              <div className="metric-item">
                <span>忠诚度</span>
                <strong>{toPercent(faithfulness)}</strong>
              </div>
              <div className="metric-item">
                <span>上下文召回率</span>
                <strong>{toPercent(contextRecall)}</strong>
              </div>
              <div className="metric-item">
                <span>答案准确度</span>
                <strong>{toPercent(answerCorrectness)}</strong>
              </div>
              <div className="metric-item">
                <span>平均 token</span>
                <strong>{toNumber(avgToken)}</strong>
              </div>
              <div className="metric-item">
                <span>平均延迟</span>
                <strong>{toNumber(avgLatency)} ms</strong>
              </div>
            </div>
            <div className="artifact-list">
              <div>汇总文件: <code>{activeJob.outputs?.summary_path || '--'}</code></div>
              <div>明细文件: <code>{activeJob.outputs?.detail_path || '--'}</code></div>
              <div>报告文件: <code>{activeJob.outputs?.report_path || '--'}</code></div>
              <div>复盘文件: <code>{activeJob.outputs?.analysis_path || '--'}</code></div>
            </div>
          </section>

          <section className="panel-card">
            <h3>失败题定位</h3>
            {failures.length === 0 ? (
              <div className="empty-state">本轮没有失败题。</div>
            ) : (
              <div className="failures-list">
                {failures.map((f) => (
                  <div key={f.id} className="failure-item">
                    <div className="failure-head">
                      <strong>{f.id}</strong>
                      <span className="bad">
                        忠诚度: {toPercent(typeof f.忠诚度 === 'number' ? f.忠诚度 : undefined)} / 召回率:{' '}
                        {toPercent(typeof f.上下文召回率 === 'number' ? f.上下文召回率 : undefined)} / 准确度:{' '}
                        {toPercent(typeof f.答案准确度 === 'number' ? f.答案准确度 : undefined)}
                      </span>
                    </div>
                    <div className="failure-question">{f.问题 || '--'}</div>
                    {f.错误 && <div className="failure-error">{f.错误}</div>}
                    <div className="failure-meta">期望证据: {(f.期望证据文件 || []).join(', ') || '--'}</div>
                    <div className="failure-meta">回答引用: {(f.引用证据文件 || []).join(', ') || '--'}</div>
                    <div className="failure-meta">检索命中: {(f.检索证据文件 || []).join(', ') || '--'}</div>
                    <div className="failure-preview">{f.答案预览 || '--'}</div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel-card">
            <h3>自动复盘报告（LLM）</h3>
            {activeJob.analysis_markdown ? (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {activeJob.analysis_markdown}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="empty-state">本轮未生成 LLM 复盘。</div>
            )}
          </section>

          <section className="panel-card">
            <h3>完整评测报告</h3>
            {activeJob.report_markdown ? (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {activeJob.report_markdown}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="empty-state">暂无报告内容。</div>
            )}
          </section>
        </>
      )}

      <section className="panel-card">
        <h3>历史运行</h3>
        {summaries.length === 0 ? (
          <div className="empty-state">暂无历史评测结果。</div>
        ) : (
          <div className="history-table">
            {summaries.map((s) => (
              <div key={s.summary_path} className="history-row">
                <div className="history-main">
                  <strong>{s.run_name}</strong>
                  <span>{s.created_at ? new Date(s.created_at).toLocaleString() : '--'}</span>
                </div>
                <div className="history-metrics">
                  <span>忠诚度: {toPercent(s.faithfulness)}</span>
                  <span>召回率: {toPercent(s.context_recall ?? s.evidence_hit_rate)}</span>
                  <span>准确度: {toPercent(s.answer_correctness ?? s.accuracy)}</span>
                  <span>token: {toNumber(s.avg_token)}</span>
                  <span>延迟: {toNumber(s.avg_latency_ms)}ms</span>
                </div>
                <button onClick={() => handleUseAsCompare(s.summary_path)}>设为对比基线</button>
              </div>
            ))}
          </div>
        )}
      </section>

      {activeJobRunning && (
        <div className="polling-tip">
          <Loader2 className="spinner" size={14} />
          正在后台轮询任务状态...
        </div>
      )}
    </div>
  );
};

export default EvaluationPanel;
