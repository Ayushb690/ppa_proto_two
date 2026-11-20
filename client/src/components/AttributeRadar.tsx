import React from 'react';
import { Compass, Sliders, ShieldCheck } from 'lucide-react';
import { AttributeScore } from '../types';

interface AttributeRadarProps {
  aestheticAttributes: AttributeScore[];
  qualityAttributes: AttributeScore[];
}

export const AttributeRadar: React.FC<AttributeRadarProps> = ({
  aestheticAttributes,
  qualityAttributes,
}) => {
  // SVG Radar Chart Math
  const size = 320;
  const center = size / 2;
  const radius = center - 52;
  const total = Math.max(1, aestheticAttributes.length);

  const getCoordinates = (index: number, value: number) => {
    const angle = (Math.PI * 2 / total) * index - Math.PI / 2;
    const dist = (value / 100) * radius;
    const x = center + dist * Math.cos(angle);
    const y = center + dist * Math.sin(angle);
    return { x, y };
  };

  // Build polygon points for current scores
  const polygonPoints = aestheticAttributes
    .map((attr, i) => {
      const { x, y } = getCoordinates(i, attr.score);
      return `${x},${y}`;
    })
    .join(' ');

  // Grid levels (25%, 50%, 75%, 100%)
  const gridLevels = [0.25, 0.5, 0.75, 1.0];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Aesthetic Radar Chart */}
      <div className="lg:col-span-7 glass-panel rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <Compass className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Aesthetic Attribute Fingerprint</h3>
              <p className="text-[11px] text-slate-400 font-mono">
                Multidimensional composition & lighting evaluation
              </p>
            </div>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-md font-mono bg-cyan-950/60 text-cyan-400 border border-cyan-800/60">
            {aestheticAttributes.length} Attributes
          </span>
        </div>

        <div className="flex items-center justify-center py-2">
          <svg className="w-full max-w-[340px] aspect-square overflow-visible" viewBox={`0 0 ${size} ${size}`}>

            {/* Concentric grid polygons */}
            {gridLevels.map((lvl, idx) => {
              const pts = Array.from({ length: total })
                .map((_, i) => {
                  const { x, y } = getCoordinates(i, lvl * 100);
                  return `${x},${y}`;
                })
                .join(' ');
              return (
                <polygon
                  key={idx}
                  points={pts}
                  fill="none"
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeWidth="1"
                  strokeDasharray={idx < 3 ? '2 2' : 'none'}
                />
              );
            })}

            {/* Axis spokes */}
            {aestheticAttributes.map((_, i) => {
              const { x, y } = getCoordinates(i, 100);
              return (
                <line
                  key={i}
                  x1={center}
                  y1={center}
                  x2={x}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.1)"
                  strokeWidth="1"
                />
              );
            })}

            {/* Filled data polygon */}
            <polygon
              points={polygonPoints}
              fill="rgba(6, 182, 212, 0.25)"
              stroke="#06b6d4"
              strokeWidth="2.5"
              className="transition-all duration-700 ease-out drop-shadow-[0_0_12px_rgba(6,182,212,0.5)]"
            />

            {/* Data points */}
            {aestheticAttributes.map((attr, i) => {
              const { x, y } = getCoordinates(i, attr.score);
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#06b6d4"
                  stroke="#0f172a"
                  strokeWidth="2"
                  className="transition-all duration-700"
                />
              );
            })}

            {/* Labels */}
            {aestheticAttributes.map((attr, i) => {
              const angle = (Math.PI * 2 / total) * i - Math.PI / 2;
              const labelDist = radius + 22;
              const lx = center + labelDist * Math.cos(angle);
              const ly = center + labelDist * Math.sin(angle);
              return (
                <text
                  key={i}
                  x={lx}
                  y={ly}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="fill-slate-400 text-[9px] font-mono font-medium hover:fill-cyan-300 transition-colors cursor-default"
                >
                  {attr.name}
                </text>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Technical Quality Attributes */}
      <div className="lg:col-span-5 glass-panel rounded-2xl p-6 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <ShieldCheck className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-100">Technical Image Quality</h3>
                <p className="text-[11px] text-slate-400 font-mono">
                  Optical, sensor & compression diagnostics
                </p>
              </div>
            </div>
            <span className="text-xs px-2 py-0.5 rounded-md font-mono bg-blue-950/60 text-blue-400 border border-blue-800/60">
              IQA
            </span>
          </div>

          <div className="space-y-4 pt-2">
            {qualityAttributes.map((attr) => (
              <div key={attr.name} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">{attr.name}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] text-slate-500 font-mono">
                      Weight: {attr.weight}%
                    </span>
                    <span className="font-mono font-bold text-slate-200">
                      {attr.score}%
                    </span>
                  </div>
                </div>
                <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden flex">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      attr.score >= 70
                        ? 'bg-emerald-500'
                        : attr.score >= 50
                        ? 'bg-amber-500'
                        : 'bg-rose-500'
                    }`}
                    style={{ width: `${attr.score}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-[11px] text-slate-400 flex items-start space-x-2">
          <Sliders className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <span>
            Evaluated using Supplementary Table S7 Quality Prompts across focus blur, sensor ISO noise, and optical color balance.
          </span>
        </div>
      </div>
    </div>
  );
};
