import { AnalysisResult } from './types';
import { generatePpaAnalysis } from './mockData';

const API_BASE = '/api';

export interface HealthStatus {
  online: boolean;
  device?: string;
  checkpoint?: string;
}

export async function checkBackendHealth(): Promise<HealthStatus> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error('Unhealthy');
    const data = await res.json();
    return {
      online: true,
      device: data.device || 'CUDA',
      checkpoint: data.checkpoint || 'best.pt',
    };
  } catch {
    return { online: false };
  }
}

export async function analyzeImage(file: File): Promise<{ result: AnalysisResult; isLive: boolean }> {
  // First try the real backend
  try {
    const formData = new FormData();
    formData.append('image', file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return { result: data, isLive: true };
    }
  } catch (err) {
    console.warn('Real backend request failed, falling back to simulated analysis:', err);
  }

  // Fallback simulation only when backend is offline
  await new Promise((r) => setTimeout(r, 650));
  const hash = file.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const pseudoRandomScore = 2.4 + ((hash % 240) / 100);
  const simulated = generatePpaAnalysis(pseudoRandomScore);

  return { result: simulated, isLive: false };
}

export async function compareImages(fileA: File, fileB: File): Promise<{
  resultA: AnalysisResult;
  resultB: AnalysisResult;
  isAWinner: boolean;
  winner: string;
  scoreDiff: number;
  isLive: boolean;
}> {
  try {
    const formData = new FormData();
    formData.append('imageA', fileA);
    formData.append('imageB', fileB);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);
    const res = await fetch(`${API_BASE}/compare`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return { ...data, isLive: true };
    }
  } catch (err) {
    console.warn('Real backend compare failed, falling back to individual analysis:', err);
  }

  const resA = await analyzeImage(fileA);
  const resB = await analyzeImage(fileB);
  const isAWinner = resA.result.rawScore >= resB.result.rawScore;
  const scoreDiff = Math.abs(resA.result.rawScore - resB.result.rawScore);

  return {
    resultA: resA.result,
    resultB: resB.result,
    isAWinner,
    winner: isAWinner ? 'Photo A' : 'Photo B',
    scoreDiff: Number(scoreDiff.toFixed(2)),
    isLive: resA.isLive && resB.isLive,
  };
}

