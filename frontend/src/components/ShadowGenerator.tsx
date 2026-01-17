import { useState, useEffect, useCallback } from 'react';
import ImageUploader from './ImageUploader';
import LightControls from './LightControls';
import PreviewCanvas from './PreviewCanvas';
import { generateShadow } from '../services/api';
import { ImageFile, ShadowParams } from '../types';

export default function ShadowGenerator() {
  const [foreground, setForeground] = useState<ImageFile | null>(null);
  const [background, setBackground] = useState<ImageFile | null>(null);
  const [params, setParams] = useState<ShadowParams>({
    lightAngle: 45,
    lightElevation: 45,
  });
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateShadowImage = useCallback(async () => {
    if (!foreground || !background) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const blob = await generateShadow({
        foreground: foreground.file,
        background: background.file,
        lightAngle: params.lightAngle,
        lightElevation: params.lightElevation,
      });

      // Create object URL for the result
      const url = URL.createObjectURL(blob);
      
      // Clean up previous URL if exists
      if (resultUrl) {
        URL.revokeObjectURL(resultUrl);
      }
      
      setResultUrl(url);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate shadow';
      setError(errorMessage);
      console.error('Error generating shadow:', err);
    } finally {
      setIsLoading(false);
    }
  }, [foreground, background, params, resultUrl]);

  // Debounced effect to regenerate shadow when inputs change
  useEffect(() => {
    if (!foreground || !background) {
      return;
    }

    const timeoutId = setTimeout(() => {
      generateShadowImage();
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [foreground, background, params, generateShadowImage]);

  const handleDownload = () => {
    if (!resultUrl) return;

    const link = document.createElement('a');
    link.href = resultUrl;
    link.download = 'shadow-result.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
          />
        </div>
      </div>
    </div>
  );
}
