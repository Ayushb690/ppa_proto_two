import React, { useState } from 'react';
import { Search, Filter, ArrowUpDown } from 'lucide-react';
import { PromptItem } from '../types';

interface PromptTableProps {
  prompts: PromptItem[];
}

export const PromptTable: React.FC<PromptTableProps> = ({ prompts }) => {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'impact' | 'score' | 'weight'>('impact');

  const filtered = prompts
    .filter((p) => {
      const matchesSearch =
        p.posPrompt.toLowerCase().includes(search.toLowerCase()) ||
        p.negPrompt.toLowerCase().includes(search.toLowerCase()) ||
        (p.attribute && p.attribute.toLowerCase().includes(search.toLowerCase()));

      const matchesCat =
        selectedCategory === 'all' || p.category === selectedCategory;

      return matchesSearch && matchesCat;
    })
    .sort((a, b) => {
      if (sortBy === 'impact') return b.impact - a.impact;
      if (sortBy === 'score') return b.score - a.score;
      return b.selectionWeight - a.selectionWeight;
    });

  return (
    <div className="glass-panel rounded-2xl p-6 space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-white">Full Prompt Pool Analysis</h3>
          <p className="text-xs text-slate-400 font-mono">
            Examining 260 antonym prompt pairs (CVPR 2026 Supp. Tables S6–S9)
          </p>
        </div>

        {/* Search & Filter */}
        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
          <div className="relative flex-1 sm:w-64">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="all">All Categories</option>
            <option value="aesthetic_attribute">Aesthetic Attributes (S6)</option>
            <option value="quality_attribute">Quality Attributes (S7)</option>
            <option value="aesthetic_general">General Aesthetic (S8)</option>
            <option value="quality_general">General Quality (S9)</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="impact">Sort by Net Impact</option>
            <option value="weight">Sort by Selection Weight (p_φ)</option>
            <option value="score">Sort by Score (s̄)</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-900/30">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase tracking-wider text-[10px] border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Positive Prompt (+)</th>
              <th className="py-3 px-4">Negative Antonym (-)</th>
              <th className="py-3 px-4">Category</th>
              <th className="py-3 px-4 text-right">Score (s̄)</th>
              <th className="py-3 px-4 text-right">Weight (p_φ)</th>
              <th className="py-3 px-4 text-right">Impact</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {filtered.map((item) => (
              <tr key={item.id} className="hover:bg-slate-800/40 transition-colors">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-200">
                  {item.posPrompt}
                </td>
                <td className="py-2.5 px-4 font-sans text-slate-400">
                  {item.negPrompt}
                </td>
                <td className="py-2.5 px-4">
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
                    {item.attribute || item.category.replace('_', ' ')}
                  </span>
                </td>
                <td className="py-2.5 px-4 text-right font-bold text-slate-100">
                  <span
                    className={
                      item.score >= 0.7
                        ? 'text-emerald-400'
                        : item.score >= 0.4
                        ? 'text-amber-400'
                        : 'text-rose-400'
                    }
                  >
                    {(item.score * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="py-2.5 px-4 text-right text-slate-300">
                  {(item.selectionWeight * 100).toFixed(2)}%
                </td>
                <td className="py-2.5 px-4 text-right text-cyan-400 font-semibold">
                  {(item.impact * 100).toFixed(3)}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-500 font-sans">
                  No prompt pairs found matching "{search}".
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
        <span>Showing {filtered.length} of {prompts.length} prompt pairs</span>
        <span>Marginalization: p(s | x, c) = Σ_t p_θ(s | x, t) · p_φ(t | x, c)</span>
      </div>
    </div>
  );
};
