import React, { useRef, useState } from 'react';
import { UploadCloud, Sparkles, RefreshCw, X, FolderOpen } from 'lucide-react';
import { SAMPLE_IMAGES } from '../mockData';
import { SampleImage } from '../types';

interface DropzoneProps {
  currentImageUrl: string | null;
  onImageSelected: (file: File | null, url: string | null, seedScore?: number, sampleId?: string) => void;
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
    onImageSelected(null, sample.url, sample.expectedScore, sample.id);
  };

  const handleClearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelected(null, null);
  };

  return (
    <div className="space-y-4">
      {/* Upload & Drop Box */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !currentImageUrl && fileInputRef.current?.click()}
        className={`relative group overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 ${isDragOver
          ? 'border-cyan-400 bg-cyan-950/20 scale-[1.01]'
          : currentImageUrl
            ? 'border-neutral-800 bg-black/60'
            : 'border-slate-300 dark:border-neutral-800 hover:border-cyan-500 dark:hover:border-neutral-700 bg-slate-50 dark:bg-neutral-950/60 hover:bg-slate-100 dark:hover:bg-neutral-900/60 cursor-pointer'
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
          <div className="relative aspect-[16/10] sm:aspect-[16/9] w-full max-h-[420px] flex items-center justify-center bg-black/80 overflow-hidden">
            <img
              src={currentImageUrl}
              alt="Evaluated subject"
              className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-[1.02]"
            />

            {/* Top Action Header on Evaluated Image */}
            <div className="absolute top-3 right-3 flex items-center space-x-2 z-20">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-3 py-1.5 rounded-xl bg-black/85 hover:bg-neutral-900 text-white border border-neutral-700/80 text-xs font-mono flex items-center space-x-1.5 backdrop-blur-md shadow-lg transition-all cursor-pointer"
                title="Upload or drop another photo"
              >
                <UploadCloud className="w-3.5 h-3.5 text-cyan-400" />
                <span>Upload New</span>
              </button>
              <button
                type="button"
                onClick={handleClearImage}
                className="p-1.5 rounded-xl bg-black/85 hover:bg-rose-950/80 hover:border-rose-700 text-neutral-300 hover:text-rose-400 border border-neutral-700/80 text-xs backdrop-blur-md shadow-lg transition-all cursor-pointer"
                title="Clear photo and show drop/upload area"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Bottom Hover Tip */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-4 cursor-pointer"
            >
              <span className="text-xs font-mono text-cyan-300 bg-black/90 px-3 py-1.5 rounded-lg border border-cyan-800/50 flex items-center space-x-2">
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Click or drop another image to replace</span>
              </span>
            </div>

            {/* Loading Overlay */}
            {isLoading && (
              <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center space-y-3 z-30">
                <div className="w-10 h-10 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin" />
                <p className="text-xs font-mono text-cyan-400 animate-pulse">Running PPA inference...</p>
              </div>
            )}
          </div>
        ) : (
          /* Interpretable Drop / Upload Symbol & Instructions */
          <div className="p-10 sm:p-14 text-center flex flex-col items-center justify-center">
            <div className="w-20 h-20 rounded-3xl bg-cyan-500/10 border-2 border-dashed border-cyan-500/40 flex items-center justify-center mb-4 group-hover:scale-110 group-hover:border-cyan-400 transition-all duration-300 shadow-lg shadow-cyan-500/5">
              <UploadCloud className="w-10 h-10 text-cyan-400" />
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white mb-2">
              Drop an image here, or <span className="text-cyan-500 dark:text-cyan-400 underline underline-offset-4">browse files</span>
            </h3>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-neutral-400 max-w-md mb-6 leading-relaxed">
              Drag & drop any photograph to evaluate its aesthetic score, lighting, composition, and technical quality.
            </p>

            <div className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-all shadow-md shadow-cyan-500/20 cursor-pointer">
              <FolderOpen className="w-4 h-4" />
              <span>Choose Photo to Evaluate</span>
            </div>

            <div className="flex items-center space-x-3 text-[11px] font-mono text-slate-400 dark:text-neutral-500 mt-6 pt-4 border-t border-slate-200 dark:border-neutral-800/80">
              <span>Supports JPEG, PNG, WebP</span>
            </div>
          </div>
        )}
      </div>

      {/* Preset Test Gallery */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-slate-500 dark:text-neutral-400 uppercase tracking-wider flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Or test with curated benchmarks</span>
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {SAMPLE_IMAGES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSelectSample(sample)}
              className="flex items-center space-x-2.5 p-2 rounded-xl border border-slate-200 dark:border-neutral-800/80 bg-white dark:bg-neutral-950/60 hover:bg-slate-50 dark:hover:bg-neutral-900 hover:border-slate-300 dark:hover:border-neutral-700 text-left transition-all group cursor-pointer shadow-sm"
            >
              <img
                src={sample.url}
                alt={sample.title}
                className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-neutral-800 group-hover:border-cyan-500/50 transition-colors"
              />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-slate-900 dark:text-slate-200 truncate group-hover:text-cyan-500 dark:group-hover:text-cyan-400 transition-colors">
                  {sample.title}
                </p>
                <p className="text-[10px] text-slate-500 dark:text-neutral-500 truncate">{sample.category}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
