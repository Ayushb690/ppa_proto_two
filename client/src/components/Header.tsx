import React from 'react';
import { Aperture, Image as ImageIcon, GitCompare, Database, Sun, Moon } from 'lucide-react';
import { HealthStatus } from '../api';

interface HeaderProps {
  activeTab: 'single' | 'compare' | 'database';
  setActiveTab: (tab: 'single' | 'compare' | 'database') => void;
  health: HealthStatus;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, health, theme, setTheme }) => {
  return (
    <header className="border-b border-slate-200 dark:border-neutral-900 bg-white/90 dark:bg-black/90 backdrop-blur-md sticky top-0 z-50 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo: Realistic Camera Lens Aperture */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-b from-neutral-800 via-neutral-900 to-black p-0.5 border border-neutral-700/60 shadow-lg shadow-black/80 flex items-center justify-center group cursor-pointer">
              <div className="w-full h-full rounded-[10px] bg-black flex items-center justify-center border border-neutral-800/80 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-cyan-500/10 to-transparent pointer-events-none" />
                <Aperture className="w-5 h-5 text-neutral-100 group-hover:text-cyan-400 group-hover:rotate-45 transition-all duration-500 relative z-10" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-slate-900 dark:text-white tracking-tight">PPA Studio</span>
              </div>
              <p className="text-xs text-slate-500 dark:text-neutral-400 hidden sm:block">
                Probabilistic Prompt Adaptation · Unified Image Aesthetics & Quality
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center space-x-1 bg-slate-100 dark:bg-neutral-900/90 p-1 rounded-xl border border-slate-200 dark:border-neutral-800">
            <button
              onClick={() => setActiveTab('single')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'single'
                  ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md shadow-cyan-500/25'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
              }`}
            >
              <ImageIcon className="w-4 h-4" />
              <span>Image Analysis</span>
            </button>
            <button
              onClick={() => setActiveTab('compare')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'compare'
                  ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md shadow-cyan-500/25'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
              }`}
            >
              <GitCompare className="w-4 h-4" />
              <span>A/B Compare</span>
            </button>
            <button
              onClick={() => setActiveTab('database')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'database'
                  ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md shadow-cyan-500/25'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800/60'
              }`}
            >
              <Database className="w-4 h-4" />
              <span>Prompt Pool (260)</span>
            </button>
          </div>

          {/* Status & Theme Toggle */}
          <div className="flex items-center space-x-2.5">
            {/* Light / Dark Mode Toggle Button */}
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-xl border border-slate-200 dark:border-neutral-800 bg-slate-100 dark:bg-neutral-900/80 hover:bg-slate-200 dark:hover:bg-neutral-800 text-slate-700 dark:text-neutral-300 transition-all shadow-sm flex items-center justify-center cursor-pointer"
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4 text-amber-400 hover:rotate-45 transition-transform" />
              ) : (
                <Moon className="w-4 h-4 text-slate-700 hover:-rotate-12 transition-transform" />
              )}
            </button>

            {/* Health Status Pill */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs border ${
                health.online
                  ? 'bg-emerald-100 dark:bg-emerald-950/60 border-emerald-300 dark:border-emerald-800 text-emerald-800 dark:text-emerald-400'
                  : 'bg-slate-100 dark:bg-slate-900 border-slate-300 dark:border-slate-800 text-slate-600 dark:text-slate-400'
              }`}
              title={health.online ? `API Server Live (${health.device})` : 'API Server Offline'}
            >
              <span className={`w-2 h-2 rounded-full ${health.online ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400 dark:bg-slate-500'}`} />
              <span className="hidden md:inline font-mono">
                {health.online ? `API: ${health.device || 'Live'}` : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
