import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { ImagePlus, RefreshCw, Save, Upload } from 'lucide-react';
import axios from 'axios';
import { imageApi } from '../api';

import './ImageIngestionPanel.css';

type ImageIngestionPanelProps = {
  provider: string;
};

const nowIsoLocal = () => new Date().toISOString();

const revokeObjectUrl = (url: string) => {
  if (!url) return;
  try {
    URL.revokeObjectURL(url);
  } catch {
    // ignore revoke failures
  }
};

const parseApiError = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as any)?.detail;
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

function ImageIngestionPanel({ provider }: ImageIngestionPanelProps) {
  const [author, setAuthor] = useState('Unknown');
  const [source, setSource] = useState('Manual image upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');

  const [isDrafting, setIsDrafting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [ocrText, setOcrText] = useState('');
  const [visualSummary, setVisualSummary] = useState('');
  const [visualDescription, setVisualDescription] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [retrievalKeywords, setRetrievalKeywords] = useState<string[]>([]);
  const [imageMeta, setImageMeta] = useState('');

  const [finalOcrText, setFinalOcrText] = useState('');
  const [finalSummary, setFinalSummary] = useState('');
  const [finalDescription, setFinalDescription] = useState('');
  const [finalTags, setFinalTags] = useState('');
  const [finalRetrievalKeywords, setFinalRetrievalKeywords] = useState('');

  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const hasDraft = useMemo(() => {
    return Boolean(visualSummary || visualDescription || ocrText);
  }, [ocrText, visualSummary, visualDescription]);

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setStatusMessage('');
    setErrorMessage('');

    revokeObjectUrl(previewUrl);
    if (!file) {
      setPreviewUrl('');
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDraft = async () => {
    if (!selectedFile) {
      setErrorMessage('请先选择图片文件。');
      return;
    }
    setIsDrafting(true);
    setErrorMessage('');
    setStatusMessage('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('author', author.trim() || 'Unknown');
      formData.append('source', source.trim() || 'Manual image upload');
      if (provider) {
        formData.append('provider', provider);
      }

      const draft = await imageApi.draft(formData);
      setOcrText(draft.ocr_text || '');
      setVisualSummary(draft.visual_summary || '');
      setVisualDescription(draft.visual_description || '');
      setTags(draft.tags || []);
      setRetrievalKeywords(draft.retrieval_keywords || []);
      setImageMeta(`${draft.image_width}x${draft.image_height}, OCR lines: ${draft.ocr_line_count}`);

      setFinalOcrText(draft.ocr_text || '');
      setFinalSummary(draft.visual_summary || '');
      setFinalDescription(draft.visual_description || '');
      setFinalTags((draft.tags || []).join(', '));
      setFinalRetrievalKeywords((draft.retrieval_keywords || []).join(', '));
    } catch (error) {
      setErrorMessage(parseApiError(error, 'Failed to run image draft.'));
    } finally {
      setIsDrafting(false);
    }
  };

  const handleUseLatestDraft = () => {
    setFinalOcrText(ocrText);
    setFinalSummary(visualSummary);
    setFinalDescription(visualDescription);
    setFinalTags(tags.join(', '));
    setFinalRetrievalKeywords(retrievalKeywords.join(', '));
  };

  const handleClear = () => {
    revokeObjectUrl(previewUrl);
    setSelectedFile(null);
    setPreviewUrl('');
    setOcrText('');
    setVisualSummary('');
    setVisualDescription('');
    setTags([]);
    setRetrievalKeywords([]);
    setImageMeta('');
    setFinalOcrText('');
    setFinalSummary('');
    setFinalDescription('');
    setFinalTags('');
    setFinalRetrievalKeywords('');
    setStatusMessage('');
    setErrorMessage('');
  };

  useEffect(() => {
    return () => {
      revokeObjectUrl(previewUrl);
    };
  }, [previewUrl]);

  const handleIngest = async () => {
    if (!selectedFile) {
      setErrorMessage('请先选择图片文件。');
      return;
    }
    if (!finalSummary.trim()) {
      setErrorMessage('请先确认视觉摘要。');
      return;
    }
    if (!finalDescription.trim()) {
      setErrorMessage('请先确认视觉描述。');
      return;
    }

    setIsSaving(true);
    setErrorMessage('');
    setStatusMessage('');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('visual_summary', finalSummary.trim());
      formData.append('visual_description', finalDescription.trim());
      formData.append('ocr_text', finalOcrText.trim());
      formData.append('tags', finalTags.trim());
      formData.append('retrieval_keywords', finalRetrievalKeywords.trim());
      formData.append('author', author.trim() || 'Unknown');
      formData.append('source', source.trim() || 'Manual image upload');
      formData.append('occurred_at', nowIsoLocal());

      const saved = await imageApi.ingest(formData);
      setStatusMessage(`已写入图片笔记：${saved.file_path}`);
    } catch (error) {
      setErrorMessage(parseApiError(error, 'Failed to ingest image note.'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="image-page" translate="no">
      <div className="image-card image-doc">
        <h2>图片资料入库（OCR + 视觉理解）</h2>
        <p>流程：上传图片 → OCR 提取文本 → 视觉模型生成摘要/描述 → 人工确认后入库。</p>
      </div>

      <div className="image-card image-controls">
        <div className="image-field">
          <label htmlFor="image-author">作者</label>
          <input
            id="image-author"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="例如：Alice"
          />
        </div>
        <div className="image-field">
          <label htmlFor="image-source">来源</label>
          <input
            id="image-source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="例如：产品截图 / PPT 导出"
          />
        </div>
        <div className="image-file">
          <label className="image-upload" htmlFor="image-file-input">
            <Upload size={16} />
            <span>{selectedFile ? selectedFile.name : '选择图片文件'}</span>
          </label>
          <input
            id="image-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
          />
        </div>
        <div className="image-buttons">
          <button className="image-action primary" onClick={handleDraft} disabled={!selectedFile || isDrafting} type="button">
            <ImagePlus size={16} />
            <span>{isDrafting ? '识别中...' : '识别并生成草稿'}</span>
          </button>
          <button className="image-action secondary" onClick={handleUseLatestDraft} disabled={!hasDraft} type="button">
            <RefreshCw size={16} />
            <span>使用最新草稿</span>
          </button>
          <button className="image-action secondary" onClick={handleClear} type="button">
            清空
          </button>
        </div>
      </div>

      <div className="image-grid">
        <div className="image-card image-preview">
          <h3>图片预览</h3>
          {previewUrl ? (
            <img src={previewUrl} alt="uploaded preview" />
          ) : (
            <div className="image-placeholder">请先选择图片文件</div>
          )}
          {imageMeta && <div className="image-meta">{imageMeta}</div>}
        </div>

        <div className="image-card">
          <h3>AI 草稿（只读）</h3>
          <label className="image-sub-label">视觉摘要</label>
          <textarea value={visualSummary} readOnly placeholder="visual summary..." />
          <label className="image-sub-label">视觉描述</label>
          <textarea value={visualDescription} readOnly placeholder="visual description..." />
          <label className="image-sub-label">OCR 文本</label>
          <textarea value={ocrText} readOnly placeholder="ocr text..." />
          <label className="image-sub-label">Tags</label>
          <input value={tags.join(', ')} readOnly placeholder="tag1, tag2..." />
          <label className="image-sub-label">检索关键词（LLM 生成）</label>
          <input value={retrievalKeywords.join(', ')} readOnly placeholder="关键词..." />
        </div>

        <div className="image-card">
          <h3>确认入库（可编辑）</h3>
          <label className="image-sub-label">确认摘要</label>
          <textarea
            value={finalSummary}
            onChange={(e) => setFinalSummary(e.target.value)}
            placeholder="请确认摘要后再入库..."
          />
          <label className="image-sub-label">确认描述</label>
          <textarea
            value={finalDescription}
            onChange={(e) => setFinalDescription(e.target.value)}
            placeholder="请确认描述后再入库..."
          />
          <label className="image-sub-label">确认 OCR 文本</label>
          <textarea
            value={finalOcrText}
            onChange={(e) => setFinalOcrText(e.target.value)}
            placeholder="可根据原图修正 OCR 文本..."
          />
          <label className="image-sub-label">确认 Tags（逗号分隔）</label>
          <input
            value={finalTags}
            onChange={(e) => setFinalTags(e.target.value)}
            placeholder="tag1, tag2..."
          />
          <label className="image-sub-label">确认检索关键词（逗号分隔）</label>
          <input
            value={finalRetrievalKeywords}
            onChange={(e) => setFinalRetrievalKeywords(e.target.value)}
            placeholder="中文+English 关键词..."
          />
          <div className="image-confirm-actions">
            <button className="image-action primary" onClick={handleIngest} disabled={isSaving || !selectedFile} type="button">
              <Save size={16} />
              <span>{isSaving ? '入库中...' : '确认入库'}</span>
            </button>
          </div>
        </div>
      </div>

      {statusMessage && <div className="image-status ok">{statusMessage}</div>}
      {errorMessage && <div className="image-status error">{errorMessage}</div>}
    </div>
  );
}

export default ImageIngestionPanel;
