interface DebugImagesViewerProps {
  shadowOnlyUrl: string | null;
  maskDebugUrl: string | null;
}

export default function DebugImagesViewer({
  shadowOnlyUrl,
  maskDebugUrl,
}: DebugImagesViewerProps) {
  if (!shadowOnlyUrl && !maskDebugUrl) {
    return null;
  }

  return (
    <div className="debug-images-viewer">
      <h3>Debug Images</h3>
      <div className="debug-images-grid">
        {shadowOnlyUrl && (
          <div className="debug-image-item">
            <h4>🖤 Shadow Only</h4>
            <div className="debug-image-container">
              <img src={shadowOnlyUrl} alt="Shadow only" />
            </div>
          </div>
        )}
        {maskDebugUrl && (
          <div className="debug-image-item">
            <h4>✂️ Mask Debug</h4>
            <div className="debug-image-container">
              <img src={maskDebugUrl} alt="Mask debug" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
