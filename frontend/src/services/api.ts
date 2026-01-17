import axios from 'axios';
import JSZip from 'jszip';

// Use relative path to leverage Vite proxy, or full URL if VITE_API_URL is set
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export interface GenerateShadowRequest {
  foreground: File;
  background: File;
  lightAngle: number;
  lightElevation: number;
}

export interface ShadowResult {
  composite: Blob;
  shadowOnly: Blob;
  maskDebug: Blob;
}

export const generateShadow = async (request: GenerateShadowRequest): Promise<ShadowResult> => {
  const formData = new FormData();
  formData.append('foreground', request.foreground);
  formData.append('background', request.background);
  formData.append('light_angle', request.lightAngle.toString());
  formData.append('light_elevation', request.lightElevation.toString());

  const response = await axios.post(
    `${API_BASE_URL}/generate-shadow`,
    formData,
    {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  // Extract ZIP file
  const zipBlob = response.data;
  const zip = await JSZip.loadAsync(zipBlob);
  
  // Extract all three images from the ZIP
  const composite = await zip.file('composite.png')?.async('blob');
  const shadowOnly = await zip.file('shadow_only.png')?.async('blob');
  const maskDebug = await zip.file('mask_debug.png')?.async('blob');

  if (!composite || !shadowOnly || !maskDebug) {
    throw new Error('Failed to extract images from ZIP file');
  }

  return {
    composite,
    shadowOnly,
    maskDebug,
  };
};
