import { useEffect, useEffectEvent, useMemo, useRef, useState } from 'react';
import { Mic, RefreshCw, Save } from 'lucide-react';
import axios from 'axios';
import { voiceApi } from '../api';

import './VoiceIngestionPanel.css';

type VoiceIngestionPanelProps = {
  provider: string;
};

type ApiValidationIssue = {
  msg?: string;
  [key: string]: unknown;
};

type ApiErrorBody = {
  detail?: string | ApiValidationIssue[] | ApiValidationIssue;
};

type SpeechRecognitionAlternativeLike = {
  transcript?: string;
};

type SpeechRecognitionResultLike = ArrayLike<SpeechRecognitionAlternativeLike> & {
  isFinal?: boolean;
};

type SpeechRecognitionEventLike = {
  resultIndex?: number;
  results?: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionErrorEventLike = {
  error?: string;
};

type RecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type RecognitionHostWindow = Window & typeof globalThis & {
  SpeechRecognition?: new () => RecognitionLike;
  webkitSpeechRecognition?: new () => RecognitionLike;
};

const getRecognitionCtor = (): (new () => RecognitionLike) | null => {
  const win = window as RecognitionHostWindow;
  return win.SpeechRecognition || win.webkitSpeechRecognition || null;
};

const nowIsoLocal = () => new Date().toISOString();

const parseApiError = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as ApiErrorBody | undefined)?.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      const joined = detail
        .map((item) => {
          if (typeof item === 'string') return item;
          if (item && typeof item === 'object' && 'msg' in item) return String(item.msg);
          return JSON.stringify(item);
        })
        .join('; ');
      if (joined) return joined;
    }
    if (error.message) return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
};

function VoiceIngestionPanel({ provider }: VoiceIngestionPanelProps) {
  const [author, setAuthor] = useState('Unknown');
  const [source, setSource] = useState('Realtime microphone');
  const [speechSupported, setSpeechSupported] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isDrafting, setIsDrafting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [rawTranscript, setRawTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [manualTranscript, setManualTranscript] = useState('');

  const [aiDraftText, setAiDraftText] = useState('');
  const [aiSummary, setAiSummary] = useState('');
  const [draftWarning, setDraftWarning] = useState('');

  const [finalTranscript, setFinalTranscript] = useState('');
  const [finalSummary, setFinalSummary] = useState('');

  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const recognitionRef = useRef<RecognitionLike | null>(null);
  const draftReqIdRef = useRef(0);
  const textEditedRef = useRef(false);
  const summaryEditedRef = useRef(false);

  const liveTranscript = useMemo(() => {
    const manual = manualTranscript.trim();
    if (manual) {
      return manual;
    }
    return [rawTranscript, interimTranscript].filter(Boolean).join(' ').trim();
  }, [rawTranscript, interimTranscript, manualTranscript]);

  const requestDraft = async (transcript: string, forceApply = false, silent = false) => {
    const normalized = transcript.trim();
    if (!normalized) {
      return;
    }

    const reqId = ++draftReqIdRef.current;
    if (!silent) {
      setIsDrafting(true);
    }
    setErrorMessage('');

    try {
      const result = await voiceApi.draft({
        transcript: normalized,
        author: author.trim(),
        provider: provider || undefined,
      });

      if (reqId !== draftReqIdRef.current) {
        return;
      }

      setAiDraftText(result.polished_text || '');
      setAiSummary(result.summary || '');
      setDraftWarning(result.warning || '');

      if (forceApply || !textEditedRef.current) {
        setFinalTranscript(result.polished_text || '');
      }
      if (forceApply || !summaryEditedRef.current) {
        setFinalSummary(result.summary || '');
      }
    } catch (error) {
      if (reqId !== draftReqIdRef.current) {
        return;
      }
      const detail = parseApiError(error, 'Failed to generate voice draft.');
      setErrorMessage(detail);
    } finally {
      if (!silent && reqId === draftReqIdRef.current) {
        setIsDrafting(false);
      }
    }
  };

  const requestDraftFromEffect = useEffectEvent((transcript: string) => {
    void requestDraft(transcript, false, true);
  });

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      setSpeechSupported(false);
      return;
    }

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'zh-CN';
    recognition.onstart = () => {
      setIsRecording(true);
    };

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      try {
        const results = event?.results;
        if (!results || typeof results.length !== 'number') {
          return;
        }

        const resultIndex = event?.resultIndex;
        const safeStart =
          typeof resultIndex === 'number' && Number.isInteger(resultIndex)
            ? Math.max(0, resultIndex)
            : 0;
        let finalPart = '';
        let interimPart = '';

        for (let i = safeStart; i < results.length; i += 1) {
          const current = results?.[i];
          const segment = String(current?.[0]?.transcript ?? '').trim();
          if (!segment) {
            continue;
          }
          if (current?.isFinal) {
            finalPart += ` ${segment}`;
          } else {
            interimPart += ` ${segment}`;
          }
        }

        if (finalPart.trim()) {
          setRawTranscript((prev) => `${prev} ${finalPart}`.trim());
        }
        setInterimTranscript(interimPart.trim());
      } catch (error) {
        console.error('Speech recognition result parse error:', error);
        setErrorMessage('Speech recognition parse error.');
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEventLike) => {
      setErrorMessage(`Speech recognition error: ${event?.error || 'unknown'}`);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
      setInterimTranscript('');
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.stop();
      } catch {
        // ignore stop error during unmount
      }
      recognitionRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!liveTranscript) {
      return;
    }
    const timer = window.setTimeout(() => {
      requestDraftFromEffect(liveTranscript);
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [liveTranscript, requestDraftFromEffect]);

  const handleStartRecording = () => {
    if (!recognitionRef.current) {
      return;
    }
    setManualTranscript('');
    setErrorMessage('');
    setStatusMessage('');
    try {
      recognitionRef.current.start();
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to start recording.';
      setErrorMessage(detail);
    }
  };

  const handleStopRecording = () => {
    if (!recognitionRef.current) {
      return;
    }
    try {
      recognitionRef.current.stop();
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to stop recording.';
      setErrorMessage(detail);
    } finally {
      setIsRecording(false);
    }
  };

  const handleUseLatestDraft = () => {
    textEditedRef.current = false;
    summaryEditedRef.current = false;
    setFinalTranscript(aiDraftText || liveTranscript);
    setFinalSummary(aiSummary || '');
  };

  const handleClear = () => {
    handleStopRecording();
    setRawTranscript('');
    setInterimTranscript('');
    setManualTranscript('');
    setAiDraftText('');
    setAiSummary('');
    setFinalTranscript('');
    setFinalSummary('');
    setDraftWarning('');
    setErrorMessage('');
    setStatusMessage('');
    textEditedRef.current = false;
    summaryEditedRef.current = false;
  };

  const handleIngest = async () => {
    const finalText = finalTranscript.trim();
    const finalBrief = finalSummary.trim();
    const finalAuthor = author.trim();

    if (!finalText) {
      setErrorMessage('请先确认最终转写文本。');
      return;
    }
    if (!finalBrief) {
      setErrorMessage('请先确认最终摘要。');
      return;
    }
    if (!finalAuthor) {
      setErrorMessage('请先设置作者。');
      return;
    }

    setIsSaving(true);
    setErrorMessage('');
    setStatusMessage('');
    try {
      const saved = await voiceApi.ingest({
        transcript: finalText,
        summary: finalBrief,
        author: finalAuthor,
        source: source.trim() || 'Realtime microphone',
        raw_transcript: liveTranscript,
        occurred_at: nowIsoLocal(),
      });
      setStatusMessage(`已写入知识库：${saved.file_path}`);
    } catch (error) {
      const detail = parseApiError(error, 'Failed to ingest voice note.');
      setErrorMessage(detail);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="voice-page" translate="no">
      <div className="voice-card voice-doc">
        <h2>语音资料入库（实时）</h2>
        <p>流程：实时语音转写 → LLM 生成草稿与摘要 → 人工校对 → 确认入库。</p>
      </div>
      {!speechSupported && (
        <div className="voice-status error">
          浏览器语音识别不可用，已切换为手动输入模式。你仍可输入文本并完成 AI 草稿与入库。
        </div>
      )}

      <div className="voice-card voice-controls">
        <div className="voice-field">
          <label htmlFor="voice-author">作者（谁说的）</label>
          <input
            id="voice-author"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="例如：Alice"
          />
        </div>
        <div className="voice-field">
          <label htmlFor="voice-source">来源</label>
          <input
            id="voice-source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="例如：周会 / 电话沟通"
          />
        </div>
        <div className="voice-buttons">
          <button
            className={`voice-action ${isRecording ? 'stop' : 'start'}`}
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            disabled={!speechSupported}
            type="button"
          >
            <Mic size={16} />
            <span className={`voice-rec-dot ${isRecording ? 'on' : 'off'}`} aria-hidden />
            <span>
              {!speechSupported ? '录音不可用' : isRecording ? '停止录音' : '开始录音'}
            </span>
          </button>
          <button
            className="voice-action secondary"
            onClick={() => void requestDraft(liveTranscript, true)}
            disabled={!liveTranscript || isDrafting}
            type="button"
          >
            <RefreshCw size={16} />
            <span>{isDrafting ? '生成中...' : '同步 AI 草稿'}</span>
          </button>
          <button className="voice-action secondary" onClick={handleClear} type="button">
            清空
          </button>
        </div>
      </div>

      <div className="voice-grid">
        <div className="voice-card">
          <h3>实时转写（原始）</h3>
          <textarea
            value={liveTranscript}
            readOnly={isRecording}
            onChange={(e) => {
              setManualTranscript(e.target.value);
              if (e.target.value.trim()) {
                setInterimTranscript('');
              }
            }}
            placeholder={
              speechSupported
                ? '录音后这里会实时显示转写文本，也可手动粘贴文本...'
                : '语音不可用，请在此输入或粘贴转写文本...'
            }
          />
          <label className="voice-sub-label">支持手动输入，输入后可直接同步 AI 草稿。</label>
        </div>

        <div className="voice-card">
          <h3>AI 草稿（建议）</h3>
          <textarea value={aiDraftText} readOnly placeholder="AI 将在这里生成润色文本..." />
          <label className="voice-sub-label">AI 摘要（建议）</label>
          <textarea className="voice-summary" value={aiSummary} readOnly placeholder="AI 摘要..." />
          {draftWarning && <div className="voice-warning">{draftWarning}</div>}
        </div>

        <div className="voice-card voice-confirm">
          <h3>确认入库（可编辑）</h3>
          <textarea
            value={finalTranscript}
            onChange={(e) => {
              textEditedRef.current = true;
              setFinalTranscript(e.target.value);
            }}
            placeholder="请确认并校正文本后再入库..."
          />
          <label className="voice-sub-label">确认摘要（可编辑）</label>
          <textarea
            className="voice-summary"
            value={finalSummary}
            onChange={(e) => {
              summaryEditedRef.current = true;
              setFinalSummary(e.target.value);
            }}
            placeholder="请确认摘要后再入库..."
          />
          <div className="voice-confirm-actions">
            <button
              className="voice-action secondary"
              onClick={handleUseLatestDraft}
              disabled={!aiDraftText}
              type="button"
            >
              使用最新 AI 草稿
            </button>
            <button className="voice-action primary" onClick={handleIngest} disabled={isSaving} type="button">
              <Save size={16} />
              <span>{isSaving ? '入库中...' : '确认入库'}</span>
            </button>
          </div>
        </div>
      </div>

      {statusMessage && <div className="voice-status ok">{statusMessage}</div>}
      {errorMessage && <div className="voice-status error">{errorMessage}</div>}
    </div>
  );
}

export default VoiceIngestionPanel;
