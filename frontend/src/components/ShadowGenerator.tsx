import { useState, useCallback } from 'react';
import ImageUploader from './ImageUploader';
import LightControls from './LightControls';
import PreviewCanvas from './PreviewCanvas';
import DebugImagesViewer from './DebugImagesViewer';
import { generateShadow } from '../services/api';
import { ImageFile, ShadowParams } from '../types';
import JSZip from 'jszip';

export default function ShadowGenerator() {
  const [foreground, setForeground] = useState<ImageFile | null>(null);
  const [background, setBackground] = useState<ImageFile | null>(null);
  const [params, setParams] = useState<ShadowParams>({
    lightAngle: 45,
    lightElevation: 45,
  });
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [shadowOnlyUrl, setShadowOnlyUrl] = useState<string | null>(null);
  const [maskDebugUrl, setMaskDebugUrl] = useState<string | null>(null);
  const [resultBlobs, setResultBlobs] = useState<{
    composite: Blob;
    shadowOnly: Blob;
    maskDebug: Blob;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateShadowImage = useCallback(async () => {
    if (!foreground || !background) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await generateShadow({
        foreground: foreground.file,
        background: background.file,
        lightAngle: params.lightAngle,
        lightElevation: params.lightElevation,
      });

      // Store the blobs for ZIP download
      setResultBlobs(result);

      // Create object URLs for all three images
      const compositeUrl = URL.createObjectURL(result.composite);
      const shadowOnlyUrl = URL.createObjectURL(result.shadowOnly);
      const maskDebugUrl = URL.createObjectURL(result.maskDebug);
      
      // Clean up previous URLs if they exist
      setResultUrl((prevUrl) => {
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl);
        }
        return compositeUrl;
      });
      setShadowOnlyUrl((prevUrl) => {
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl);
        }
        return shadowOnlyUrl;
      });
      setMaskDebugUrl((prevUrl) => {
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl);
        }
        return maskDebugUrl;
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate shadow';
      setError(errorMessage);
      console.error('Error generating shadow:', err);
    } finally {
      setIsLoading(false);
    }
  }, [foreground, background, params]);

  const handleDownload = () => {
    if (!resultUrl) return;

    // Trigger download of the composite image
    const link = document.createElement('a');
    link.href = resultUrl;
    link.download = 'composite.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadZip = async () => {
    if (!resultBlobs) return;

    try {
      // Create a new ZIP file
      const zip = new JSZip();
      
      // Add all three images to the ZIP
      zip.file('composite.png', resultBlobs.composite);
      zip.file('shadow_only.png', resultBlobs.shadowOnly);
      zip.file('mask_debug.png', resultBlobs.maskDebug);
      
      // Generate the ZIP file
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      
      // Trigger download
      const link = document.createElement('a');
      link.href = URL.createObjectURL(zipBlob);
      link.download = 'shadow_result.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the object URL
      URL.revokeObjectURL(link.href);
    } catch (err) {
      console.error('Error creating ZIP file:', err);
      setError('Failed to create ZIP file');
    }
  };

  const handleGenerate = () => {
    if (foreground && background) {
      generateShadowImage();
    }
  };

  return (
    <div className="shadow-generator">
      <header className="app-header">
        <h1>🎨 Realistic Shadow Generator</h1>
        <p>Create realistic shadows with directional light control</p>
      </header>

      <div className="generator-container">
        <div className="input-section">
          <div className="upload-section">
            <ImageUploader
              label="Foreground Image"
              image={foreground}
              onImageChange={setForeground}
            />
            <ImageUploader
              label="Background Image"
              image={background}
              onImageChange={setBackground}
            />
          </div>

          <div className="controls-section">
            <LightControls params={params} onParamsChange={setParams} />
            
            <button
              className="generate-button"
              onClick={handleGenerate}
              disabled={!foreground || !background || isLoading}
              type="button"
            >
              {isLoading ? 'Generating...' : 'Generate Shadow'}
            </button>
          </div>
        </div>

        <div className="preview-section">
          <PreviewCanvas
            imageUrl={resultUrl}
            isLoading={isLoading}
            error={error}
            onDownload={handleDownload}
            onDownloadZip={handleDownloadZip}
            hasResult={!!resultUrl}
          />
          {resultUrl && (
            <DebugImagesViewer
              shadowOnlyUrl={shadowOnlyUrl}
              maskDebugUrl={maskDebugUrl}
            />
          )}
        </div>
      </div>
    </div>
  );
}
