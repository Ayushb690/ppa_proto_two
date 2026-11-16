import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Dropzone } from './components/Dropzone';
import { ScoreGauge } from './components/ScoreGauge';
import { ExplainabilityPanel } from './components/ExplainabilityPanel';
import { AttributeRadar } from './components/AttributeRadar';
import { PromptTable } from './components/PromptTable';
import { CompareView } from './components/CompareView';
import { SAMPLE_IMAGES, generatePpaAnalysis } from './mockData';
import { AnalysisResult } from './types';
import { checkBackendHealth, analyzeImage, HealthStatus } from './api';
import { Sparkles, ArrowRight, ShieldCheck, Cpu } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'single' | 'compare' | 'database'>('single');
  const [health, setHealth] = useState<HealthStatus>({ online: false });
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(SAMPLE_IMAGES[0].url);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult>(
    generatePpaAnalysis(SAMPLE_IMAGES[0].expectedScore)
  );
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Poll backend health once on mount
  useEffect(() => {
    checkBackendHealth().then(setHealth);
  }, []);

  const handleImageSelected = async (file: File | null, url: string | null, seedScore?: number) => {
    if (!url) return;
    setCurrentImageUrl(url);
    setIsLoading(true);

    try {
      if (file) {
        const { result, isLive } = await analyzeImage(file);
        setAnalysisResult(result);
        if (isLive && !health.online) {
          setHealth({ online: true, device: result.device });
        }
      } else if (seedScore !== undefined) {
        // Preset sample image selected
        await new Promise((r) => setTimeout(r, 300));
        setAnalysisResult(generatePpaAnalysis(seedScore));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500/20 selection:text-cyan-300">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Sub-header Banner */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-slate-900">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center space-x-2">
              <span>Image Aesthetics & Quality Assessment</span>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60 hidden sm:inline-block">
                Concept Bottleneck Model
              </span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Interpretable scoring powered by OpenCLIP ViT-B/16 and 260 human-readable antonym pairs.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-xl border border-slate-800 self-start md:self-auto">
            <span className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Weights: {health.online ? 'best.pt' : 'PARA Checkpoint'}</span>
            </span>
            <span>·</span>
            <span>28.68M Params</span>
          </div>
        </div>

        {/* Tab 1: Single Image Analysis */}
        {activeTab === 'single' && (
          <div className="space-y-8">
            <Dropzone
              currentImageUrl={currentImageUrl}
              onImageSelected={handleImageSelected}
              isLoading={isLoading}
            />

            {/* Score & Gauge Card */}
            <ScoreGauge result={analysisResult} />

            {/* Core Explainability Panel: Positive Drivers vs Negative Penalties */}
            <ExplainabilityPanel
              positivePrompts={analysisResult.topPositivePrompts}
              negativePrompts={analysisResult.topNegativePrompts}
            />

            {/* Multidimensional Radar & Technical IQA Breakdown */}
            <AttributeRadar
              aestheticAttributes={analysisResult.attributeBreakdown}
              qualityAttributes={analysisResult.qualityBreakdown}
            />

            {/* Bottom Callout to Full Table */}
            <div className="glass-panel rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-slate-800">
              <div>
                <h4 className="text-sm font-semibold text-white">Explore Full 260 Prompt Pool</h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Inspect every prompt pair across the 4 supplementary categories with individual scores and weights.
                </p>
              </div>
              <button
                onClick={() => setActiveTab('database')}
                className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-all flex items-center space-x-1.5 shadow-lg shadow-cyan-500/20 shrink-0"
              >
                <span>Open Prompt Inspector</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Tab 2: A/B Comparison */}
        {activeTab === 'compare' && <CompareView />}

        {/* Tab 3: Full 260 Prompt Pool Database */}
        {activeTab === 'database' && (
          <PromptTable prompts={analysisResult.allPrompts} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 mt-12 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>
            PPA: Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment (CVPR 2026)
          </p>
          <div className="flex items-center space-x-4">
            <span className="text-slate-400">PARA Benchmark (GIAA)</span>
            <span>·</span>
            <span className="text-cyan-400">SRCC: 0.9164 / PLCC: 0.9456</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
