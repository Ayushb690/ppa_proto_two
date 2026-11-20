import React from 'react';
import { Sparkles, ExternalLink, Image as ImageIcon, GitCompare, Database } from 'lucide-react';
import { HealthStatus } from '../api';

interface HeaderProps {
  activeTab: 'single' | 'compare' | 'database';
  setActiveTab: (tab: 'single' | 'compare' | 'database') => void;
  health: HealthStatus;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, health }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Paper Citation */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">PPA Studio</span>
                <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  CVPR 2026
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Probabilistic Prompt Adaptation · Unified Image Aesthetics & Quality
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('single')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'single'
                  ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <ImageIcon className="w-4 h-4" />
              <span>Image Analysis</span>
            </button>
            <button
              onClick={() => setActiveTab('compare')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'compare'
                  ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <GitCompare className="w-4 h-4" />
              <span>A/B Compare</span>
            </button>
            <button
              onClick={() => setActiveTab('database')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'database'
                  ? 'bg-cyan-500 text-white shadow-md shadow-cyan-500/25'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Database className="w-4 h-4" />
              <span>Prompt Pool (260)</span>
            </button>
          </div>

          {/* Status & Links */}
          <div className="flex items-center space-x-3">
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs border ${
                health.online
                  ? 'bg-emerald-950/60 border-emerald-800 text-emerald-400'
                  : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}
              title={health.online ? `API Server Live (${health.device})` : 'Offline (Client Simulation Active)'}
            >
              <span className={`w-2 h-2 rounded-full ${health.online ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
              <span className="hidden md:inline font-mono">
                {health.online ? `API: ${health.device || 'Live'}` : 'Demo Simulation'}
              </span>
            </div>

            <a
              href="https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf"
              target="_blank"
              rel="noreferrer"
              className="text-slate-400 hover:text-cyan-400 p-2 rounded-lg hover:bg-slate-900 transition-colors"
              title="View CVPR 2026 Paper"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </header>
  );
};
