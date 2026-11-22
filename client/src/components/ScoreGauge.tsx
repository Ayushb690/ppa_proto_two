import React from 'react';
import { Award, Zap, Layers } from 'lucide-react';
import { AnalysisResult } from '../types';

interface ScoreGaugeProps {
  result: AnalysisResult;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({ result }) => {
  // SVG circular gauge calculation
  const radius = 70;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  // Sweep an arc of 270 degrees (or full circle)
  const percent = Math.min(100, Math.max(0, result.percentile));
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <div className="glass-panel rounded-2xl p-6 relative overflow-hidden">
      {/* Background radial gradient */}
      <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
        {/* Radial SVG Gauge */}
        <div className="relative flex items-center justify-center shrink-0">
          <svg className="w-44 h-44 -rotate-90 transform" viewBox="0 0 160 160">
            {/* Background track */}
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke="currentColor"
              strokeWidth={strokeWidth}
              fill="transparent"
              className="text-slate-800"
            />
            {/* Value stroke */}
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke="url(#score-gradient)"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              className="transition-all duration-1000 ease-out"
            />
            <defs>
              <linearGradient id="score-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" />
                <stop offset="100%" stopColor="#3b82f6" />
              </linearGradient>
            </defs>
          </svg>

          {/* Center Score Numbers */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-extrabold tracking-tight text-white font-mono">
              {result.rawScore.toFixed(2)}
            </span>
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-widest">
              PARA [1 - 5]
            </span>
            <span className="text-xs font-semibold text-cyan-400 mt-0.5">
              {result.percentile}% Aesthetic
            </span>
          </div>
        </div>

        {/* Rating Breakdown & Metadata */}
        <div className="flex-1 w-full space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-1">
                Predicted Aesthetic Tier
              </p>
              <div className="flex items-center space-x-2">
                <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${result.tierColor} flex items-center space-x-1.5`}>
                  <Award className="w-3.5 h-3.5" />
                  <span>{result.tier}</span>
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  s̄ ∈ [{result.normalizedScore.toFixed(4)}]
                </span>
              </div>
            </div>

            <div className="text-right">
              <span className="text-xs font-mono text-slate-500 block">Marginalization</span>
              <span className="text-xs font-mono text-cyan-400">Σ_t p_θ · p_φ</span>
            </div>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            Overall score is calculated by evaluating the image against 260 human-interpretable antonym
            prompt pairs via CLIP-IQA and marginalizing with the prompt selection network.
          </p>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80">
            <div className="bg-slate-900/60 p-2 rounded-xl border border-slate-800">
              <div className="flex items-center space-x-1 text-slate-400 text-[10px] uppercase font-mono">
                <Zap className="w-3 h-3 text-amber-400" />
                <span>Latency</span>
              </div>
              <p className="text-xs font-mono font-semibold text-slate-200 mt-0.5">
                {result.inferenceTimeMs} ms
              </p>
            </div>

            <div className="bg-slate-900/60 p-2 rounded-xl border border-slate-800">
              <div className="flex items-center space-x-1 text-slate-400 text-[10px] uppercase font-mono">
                <Layers className="w-3 h-3 text-purple-400" />
                <span>Prompt Pool</span>
              </div>
              <p className="text-xs font-mono font-semibold text-slate-200 mt-0.5">
                260 pairs
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
