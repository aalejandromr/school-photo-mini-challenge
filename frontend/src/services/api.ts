import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface GenerateShadowRequest {
  foreground: File;
  background: File;
  lightAngle: number;
  lightElevation: number;
}

export const generateShadow = async (request: GenerateShadowRequest): Promise<Blob> => {
  const formData = new FormData();
  formData.append('foreground', request.foreground);
  formData.append('background', request.background);
  formData.append('light_angle', request.lightAngle.toString());
  formData.append('light_elevation', request.lightElevation.toString());

  const response = await axios.post(
    `${API_BASE_URL}/api/generate-shadow`,
    formData,
    {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};
