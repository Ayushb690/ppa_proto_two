import React from 'react';
import { CheckCircle2, AlertCircle, Info, ChevronRight } from 'lucide-react';
import { PromptItem } from '../types';

interface ExplainabilityPanelProps {
  positivePrompts: PromptItem[];
  negativePrompts: PromptItem[];
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  positivePrompts,
  negativePrompts,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Positive Drivers Card */}
      <div className="glass-panel rounded-2xl p-6 border-slate-800">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Top Aesthetic Drivers</h3>
              <p className="text-[11px] text-slate-400 font-mono">
                High selection weight (p_φ) & positive score (s̄)
              </p>
            </div>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-md font-mono bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
            Boosts Score
          </span>
        </div>

        <div className="space-y-3">
          {positivePrompts.map((p) => (
            <div
              key={p.id}
              className="p-3 rounded-xl bg-slate-900/50 border border-slate-800/80 hover:border-slate-700 transition-all space-y-2"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-emerald-400 truncate">
                      "{p.posPrompt}"
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                      vs "{p.negPrompt}"
                    </span>
                  </div>
                  {p.attribute && (
                    <span className="inline-block mt-0.5 text-[10px] font-mono text-cyan-400/90 bg-cyan-950/40 px-1.5 py-0.2 rounded border border-cyan-800/40">
                      {p.attribute}
                    </span>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <span className="text-xs font-mono font-bold text-slate-200">
                    {(p.score * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-slate-500 block font-mono">
                    w: {(p.selectionWeight * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Weight & score bar */}
              <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden flex">
                <div
                  className="bg-emerald-500 h-full rounded-full transition-all duration-700"
                  style={{ width: `${Math.min(100, p.score * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Negative Drivers / Technical Flaws Card */}
      <div className="glass-panel rounded-2xl p-6 border-slate-800">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <AlertCircle className="w-4 h-4 text-rose-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Negative Flaws & Penalties</h3>
              <p className="text-[11px] text-slate-400 font-mono">
                Prompts dragging down the evaluation
              </p>
            </div>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-md font-mono bg-rose-950/60 text-rose-400 border border-rose-800/60">
            Penalties
          </span>
        </div>

        <div className="space-y-3">
          {negativePrompts.length > 0 ? (
            negativePrompts.map((p) => (
              <div
                key={p.id}
                className="p-3 rounded-xl bg-slate-900/50 border border-slate-800/80 hover:border-slate-700 transition-all space-y-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-semibold text-rose-400 truncate">
                        "{p.negPrompt}"
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                        vs "{p.posPrompt}"
                      </span>
                    </div>
                    {p.attribute && (
                      <span className="inline-block mt-0.5 text-[10px] font-mono text-rose-400/90 bg-rose-950/40 px-1.5 py-0.2 rounded border border-rose-800/40">
                        {p.attribute}
                      </span>
                    )}
                  </div>
                  <div className="text-right shrink-0">
                    <span className="text-xs font-mono font-bold text-rose-300">
                      {(p.score * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-slate-500 block font-mono">
                      w: {(p.selectionWeight * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Score bar */}
                <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden flex">
                  <div
                    className="bg-rose-500 h-full rounded-full transition-all duration-700"
                    style={{ width: `${Math.min(100, (1 - p.score) * 100)}%` }}
                  />
                </div>
              </div>
            ))
          ) : (
            <div className="p-8 text-center text-slate-500 text-xs">
              <p>No significant technical flaws or negative penalties detected for this image.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
