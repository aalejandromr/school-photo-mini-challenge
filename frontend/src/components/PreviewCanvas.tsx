interface PreviewCanvasProps {
  imageUrl: string | null;
  isLoading: boolean;
  error: string | null;
  onDownload?: () => void;
  onDownloadZip?: () => void;
  hasResult?: boolean;
}

export default function PreviewCanvas({
  imageUrl,
  isLoading,
  error,
  onDownload,
  onDownloadZip,
  hasResult = false,
}: PreviewCanvasProps) {
  if (error) {
    return (
      <div className="preview-canvas error">
        <div className="error-message">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="preview-canvas loading">
        <div className="spinner">
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
        </div>
        <p>Generating shadow...</p>
      </div>
    );
  }

  if (!imageUrl) {
    return (
      <div className="preview-canvas empty">
        <div className="empty-message">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h6v6H9z" />
          </svg>
          <p>Preview will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="preview-canvas">
      <div className="preview-image-container">
        <img src={imageUrl} alt="Shadow preview" />
        {hasResult && (
          <div className="download-buttons">
            {onDownload && (
              <button className="download-button" onClick={onDownload} type="button">
                Download Composite
              </button>
            )}
            {onDownloadZip && (
              <button className="download-zip-button" onClick={onDownloadZip} type="button">
                Download ZIP (All 3 Images)
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
