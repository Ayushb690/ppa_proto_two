import React, { useState } from 'react';
import { GitCompare, Trophy, ArrowRight, UploadCloud, CheckCircle, Loader2 } from 'lucide-react';
import { SAMPLE_IMAGES, generatePpaAnalysis } from '../mockData';
import { AnalysisResult } from '../types';
import { analyzeImage } from '../api';

export const CompareView: React.FC = () => {
  const [imgAUrl, setImgAUrl] = useState<string>(SAMPLE_IMAGES[0].url);
  const [imgBUrl, setImgBUrl] = useState<string>(SAMPLE_IMAGES[1].url);
  const [resultA, setResultA] = useState<AnalysisResult>(generatePpaAnalysis(SAMPLE_IMAGES[0].expectedScore));
  const [resultB, setResultB] = useState<AnalysisResult>(generatePpaAnalysis(SAMPLE_IMAGES[1].expectedScore));
  const [loadingA, setLoadingA] = useState<boolean>(false);
  const [loadingB, setLoadingB] = useState<boolean>(false);

  const handleUploadA = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImgAUrl(url);
      setLoadingA(true);
      try {
        const { result } = await analyzeImage(file);
        setResultA(result);
      } finally {
        setLoadingA(false);
      }
    }
  };

  const handleUploadB = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImgBUrl(url);
      setLoadingB(true);
      try {
        const { result } = await analyzeImage(file);
        setResultB(result);
      } finally {
        setLoadingB(false);
      }
    }
  };


  const isAWinner = resultA.rawScore >= resultB.rawScore;
  const scoreDiff = Math.abs(resultA.rawScore - resultB.rawScore);

  return (
    <div className="space-y-6">
      {/* Overview & Winner Banner */}
      <div className="glass-panel rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-6 border-slate-800">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
            <Trophy className="w-6 h-6 text-amber-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-bold text-white">
                {isAWinner ? 'Photo A Wins' : 'Photo B Wins'}
              </h2>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                +{(scoreDiff).toFixed(2)} PARA Points
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Comparative aesthetic ranking evaluated via PPA concept marginalization.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-6 text-xs font-mono">
          <div className="text-right">
            <span className="text-slate-500 block">Photo A Score</span>
            <span className="text-base font-bold text-cyan-400">{resultA.rawScore.toFixed(2)}</span>
          </div>
          <span className="text-slate-600 font-bold">VS</span>
          <div className="text-left">
            <span className="text-slate-500 block">Photo B Score</span>
            <span className="text-base font-bold text-blue-400">{resultB.rawScore.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Side by side image cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Photo A */}
        <div className={`glass-panel rounded-2xl p-5 border-2 transition-all ${isAWinner ? 'border-cyan-500/80 shadow-lg shadow-cyan-500/10' : 'border-slate-800'}`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 font-mono text-xs flex items-center justify-center font-bold">
                A
              </span>
              <span className="text-sm font-semibold text-white">Subject A</span>
            </div>
            <label className="text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer flex items-center space-x-1 font-mono">
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Change Image</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleUploadA} />
            </label>
          </div>

          <div className="relative aspect-[16/10] rounded-xl overflow-hidden bg-black/60 mb-4">
            <img src={imgAUrl} alt="Subject A" className="w-full h-full object-contain" />
            {loadingA && (
              <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center space-y-2 backdrop-blur-sm">
                <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                <span className="text-xs font-mono text-cyan-300">Evaluating PPA...</span>
              </div>
            )}
            {!loadingA && isAWinner && (
              <div className="absolute top-3 right-3 bg-cyan-500 text-slate-950 px-2.5 py-1 rounded-md text-xs font-bold font-mono flex items-center space-x-1 shadow-md">
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Winner</span>
              </div>
            )}
          </div>

          {/* Key Metrics */}
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/80">
              <span className="text-slate-400">Predicted Aesthetic Score:</span>
              <span className="font-mono font-bold text-white">{resultA.rawScore.toFixed(2)} / 5.00</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/80">
              <span className="text-slate-400">Quality Classification:</span>
              <span className="font-mono text-cyan-400">{resultA.tier}</span>
            </div>
            <div className="py-2">
              <span className="text-[11px] text-slate-500 font-mono uppercase block mb-1">Top Positive Driver</span>
              <span className="text-xs text-emerald-400 font-medium">"{resultA.topPositivePrompts[0]?.posPrompt}"</span>
            </div>
          </div>
        </div>

        {/* Photo B */}
        <div className={`glass-panel rounded-2xl p-5 border-2 transition-all ${!isAWinner ? 'border-blue-500/80 shadow-lg shadow-blue-500/10' : 'border-slate-800'}`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-full bg-blue-500/20 text-blue-400 font-mono text-xs flex items-center justify-center font-bold">
                B
              </span>
              <span className="text-sm font-semibold text-white">Subject B</span>
            </div>
            <label className="text-xs text-blue-400 hover:text-blue-300 cursor-pointer flex items-center space-x-1 font-mono">
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Change Image</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleUploadB} />
            </label>
          </div>

          <div className="relative aspect-[16/10] rounded-xl overflow-hidden bg-black/60 mb-4">
            <img src={imgBUrl} alt="Subject B" className="w-full h-full object-contain" />
            {loadingB && (
              <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center space-y-2 backdrop-blur-sm">
                <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
                <span className="text-xs font-mono text-blue-300">Evaluating PPA...</span>
              </div>
            )}
            {!loadingB && !isAWinner && (
              <div className="absolute top-3 right-3 bg-blue-500 text-slate-950 px-2.5 py-1 rounded-md text-xs font-bold font-mono flex items-center space-x-1 shadow-md">
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Winner</span>
              </div>
            )}
          </div>

          {/* Key Metrics */}
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/80">
              <span className="text-slate-400">Predicted Aesthetic Score:</span>
              <span className="font-mono font-bold text-white">{resultB.rawScore.toFixed(2)} / 5.00</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/80">
              <span className="text-slate-400">Quality Classification:</span>
              <span className="font-mono text-blue-400">{resultB.tier}</span>
            </div>
            <div className="py-2">
              <span className="text-[11px] text-slate-500 font-mono uppercase block mb-1">Top Positive Driver</span>
              <span className="text-xs text-emerald-400 font-medium">"{resultB.topPositivePrompts[0]?.posPrompt}"</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
