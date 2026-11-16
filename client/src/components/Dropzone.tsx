import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, RefreshCw } from 'lucide-react';
import { SAMPLE_IMAGES } from '../mockData';
import { SampleImage } from '../types';

interface DropzoneProps {
  currentImageUrl: string | null;
  onImageSelected: (file: File | null, url: string | null, seedScore?: number) => void;
  isLoading: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({ currentImageUrl, onImageSelected, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    }
  };

  const handleSelectSample = (sample: SampleImage) => {
    onImageSelected(null, sample.url, sample.expectedScore);
  };

  return (
    <div className="space-y-4">
      {/* Upload Box */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative group cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-950/20 scale-[1.01]'
            : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={handleFileChange}
        />

        {currentImageUrl ? (
          <div className="relative aspect-[16/10] sm:aspect-[16/9] w-full max-h-[380px] flex items-center justify-center bg-black/60 overflow-hidden">
            <img
              src={currentImageUrl}
              alt="Evaluated subject"
              className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-4">
              <span className="text-xs font-mono text-cyan-300 bg-slate-900/90 px-2.5 py-1 rounded-md border border-cyan-800/50 flex items-center space-x-1.5">
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Click or drop new image to re-evaluate</span>
              </span>
            </div>
            {isLoading && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center space-y-3">
                <div className="w-10 h-10 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin" />
                <p className="text-xs font-mono text-cyan-400 animate-pulse">Running PPA inference...</p>
              </div>
            )}
          </div>
        ) : (
          <div className="p-8 sm:p-12 text-center flex flex-col items-center justify-center">
            <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <UploadCloud className="w-8 h-8 text-cyan-400" />
            </div>
            <h3 className="text-base sm:text-lg font-semibold text-white mb-1">
              Drop an image here, or <span className="text-cyan-400">browse</span>
            </h3>
            <p className="text-xs sm:text-sm text-slate-400 max-w-sm mb-4">
              Supports JPEG, PNG, or WebP. Evaluates overall aesthetics, composition, lighting, and technical flaws.
            </p>
            <div className="flex items-center space-x-4 text-xs font-mono text-slate-500">
              <span>RGB 224×224 OpenCLIP</span>
              <span>·</span>
              <span>260 Antonym Prompts</span>
            </div>
          </div>
        )}
      </div>

      {/* Preset Test Gallery */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Or test with curated benchmarks</span>
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {SAMPLE_IMAGES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSelectSample(sample)}
              className="flex items-center space-x-2.5 p-2 rounded-xl border border-slate-800/80 bg-slate-900/60 hover:bg-slate-800/80 hover:border-slate-700 text-left transition-all group"
            >
              <img
                src={sample.url}
                alt={sample.title}
                className="w-10 h-10 rounded-lg object-cover border border-slate-700 group-hover:border-cyan-500/50 transition-colors"
              />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-slate-200 truncate group-hover:text-cyan-400 transition-colors">
                  {sample.title}
                </p>
                <p className="text-[10px] text-slate-500 truncate">{sample.category}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
