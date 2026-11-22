import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Dropzone } from './components/Dropzone';
import { ScoreGauge } from './components/ScoreGauge';
import { ExplainabilityPanel } from './components/ExplainabilityPanel';
import { AttributeRadar } from './components/AttributeRadar';
import { PromptTable } from './components/PromptTable';
import { CompareView } from './components/CompareView';
import { generatePpaAnalysis } from './mockData';
import { AnalysisResult } from './types';
import { checkBackendHealth, analyzeImage, HealthStatus } from './api';
import { ArrowRight, Cpu } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'single' | 'compare' | 'database'>('single');
  const [health, setHealth] = useState<HealthStatus>({ online: false });
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('ppa_theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return 'dark';
  });
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Sync theme with HTML class and localStorage
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    }
    localStorage.setItem('ppa_theme', theme);
  }, [theme]);

  // Poll backend health once on mount
  useEffect(() => {
    checkBackendHealth().then((status) => {
      setHealth(status);
    });
  }, []);

  const handleImageSelected = async (file: File | null, url: string | null, seedScore?: number, sampleId?: string) => {
    if (!url) {
      setCurrentImageUrl(null);
      setAnalysisResult(null);
      return;
    }
    setCurrentImageUrl(url);
    setIsLoading(true);

    try {
      if (file) {
        const { result, isLive } = await analyzeImage(file);
        setAnalysisResult(result);
        if (isLive && !health.online) {
          setHealth({ online: true, device: result.device });
        }
      } else if (url && health.online) {
        // Evaluate preset sample through live model if backend is online
        try {
          const resp = await fetch(url);
          const blob = await resp.blob();
          const sampleFile = new File([blob], 'sample.jpg', { type: blob.type || 'image/jpeg' });
          const { result, isLive } = await analyzeImage(sampleFile);
          setAnalysisResult(result);
          if (isLive && !health.online) {
            setHealth({ online: true, device: result.device });
          }
        } catch {
          if (seedScore !== undefined) {
            setAnalysisResult(generatePpaAnalysis(seedScore, sampleId));
          }
        }
      } else if (seedScore !== undefined) {
        // Preset sample image selected (offline simulation fallback)
        await new Promise((r) => setTimeout(r, 300));
        setAnalysisResult(generatePpaAnalysis(seedScore, sampleId));
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-black text-slate-900 dark:text-slate-100 transition-colors duration-300 selection:bg-cyan-500/20 selection:text-cyan-600 dark:selection:text-cyan-300">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} health={health} theme={theme} setTheme={setTheme} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Sub-header Banner */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-slate-200 dark:border-neutral-900">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
              Image Aesthetics & Quality Assessment
            </h1>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-neutral-900/80 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-neutral-800 self-start md:self-auto">
            <span className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-500 dark:text-cyan-400" />
              <span>Weights: {health.online ? 'best.pt (Epoch 3)' : 'PARA Checkpoint'}</span>
            </span>
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

            {/* Score & Gauge Card and Detailed Breakdowns */}
            {analysisResult && (
              <>
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
                <div className="glass-panel rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4 border-slate-200 dark:border-neutral-800">
                  <div>
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white">Explore Full 260 Prompt Pool</h4>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      Inspect every prompt pair across the 4 supplementary categories with individual scores and weights.
                    </p>
                  </div>
                  <button
                    onClick={() => setActiveTab('database')}
                    className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-all flex items-center space-x-2 shrink-0 shadow-lg shadow-cyan-500/20 cursor-pointer"
                  >
                    <span>View Full Prompt Pool</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* Tab 2: A/B Comparison */}
        {activeTab === 'compare' && <CompareView />}

        {/* Tab 3: Full 260 Prompt Pool Database */}
        {activeTab === 'database' && (
          <PromptTable prompts={analysisResult ? analysisResult.allPrompts : generatePpaAnalysis().allPrompts} />
        )}
      </main>
    </div>
  );
}

export default App;
