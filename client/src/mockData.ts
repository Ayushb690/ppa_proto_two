import { AnalysisResult, PromptItem, SampleImage } from './types';

export const SAMPLE_IMAGES: SampleImage[] = [
  {
    id: 'golden-hour-landscape',
    title: 'Alpine Sunrise Glow',
    category: 'Landscape / Nature',
    description: 'High dynamic range, rich color temperature and clear rule-of-thirds mountain composition.',
    url: 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop&q=80',
    expectedScore: 4.45,
  },
  {
    id: 'cinematic-portrait',
    title: 'Studio Bokeh Portrait',
    category: 'Portraiture',
    description: 'Shallow depth of field, sharp subject eye focus, and balanced Rembrandt studio lighting.',
    url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&auto=format&fit=crop&q=80',
    expectedScore: 4.28,
  },
  {
    id: 'urban-architecture',
    title: 'Geometric Facade',
    category: 'Architecture',
    description: 'Strict architectural symmetry, repetitive window patterns and stark monochrome contrast.',
    url: 'https://images.unsplash.com/photo-1513694203232-719a280e022f?w=800&auto=format&fit=crop&q=80',
    expectedScore: 3.92,
  },
  {
    id: 'blurry-night-motion',
    title: 'Night Motion Blur',
    category: 'Low Quality / Artifacts',
    description: 'Severe camera shake, high ISO chroma noise, and underexposed shadow clipping.',
    url: 'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=800&auto=format&fit=crop&q=80',
    expectedScore: 1.85,
  },
];

// Curated subset of the 260 antonym prompts from src/ppa/prompts.py
export const CORE_PROMPT_DATABASE = [
  { pos: 'Good lighting.', neg: 'Poor lighting.', cat: 'aesthetic_attribute', attr: 'Good Lighting' },
  { pos: 'Harmonious colors.', neg: 'Clashing colors.', cat: 'aesthetic_attribute', attr: 'Color Harmony' },
  { pos: 'Rule of thirds composition.', neg: 'Off-balance non-standard composition.', cat: 'aesthetic_attribute', attr: 'Rule of Thirds' },
  { pos: 'Clear subject emphasis.', neg: 'Cluttered indistinct subject.', cat: 'aesthetic_attribute', attr: 'Subject Emphasis' },
  { pos: 'Pleasing shallow depth of field.', neg: 'Distracting busy background depth.', cat: 'aesthetic_attribute', attr: 'Depth of Field' },
  { pos: 'Vivid and rich colors.', neg: 'Dull washed out colors.', cat: 'aesthetic_attribute', attr: 'Vivid Colors' },
  { pos: 'Well-balanced symmetrical frame.', neg: 'Asymmetrical unbalanced frame.', cat: 'aesthetic_attribute', attr: 'Balance & Symmetry' },
  { pos: 'Intriguing visual content.', neg: 'Boring monotonous content.', cat: 'aesthetic_attribute', attr: 'Interesting Content' },
  { pos: 'Deliberate artistic motion blur.', neg: 'Accidental chaotic motion blur.', cat: 'aesthetic_attribute', attr: 'Motion & Blur' },
  { pos: 'Rhythmic repetition and pattern.', neg: 'Disordered chaotic patterns.', cat: 'aesthetic_attribute', attr: 'Repetition & Pattern' },
  
  // Quality attributes
  { pos: 'Tack sharp focus.', neg: 'Out of focus blurry image.', cat: 'quality_attribute', attr: 'Blur' },
  { pos: 'Clean image without digital noise.', neg: 'Grainy image with high sensor noise.', cat: 'quality_attribute', attr: 'Noise' },
  { pos: 'Natural true-to-life color reproduction.', neg: 'Severe color cast or tinting.', cat: 'quality_attribute', attr: 'Color-related' },
  { pos: 'Crisp micro-contrast.', neg: 'Muddy low contrast.', cat: 'quality_attribute', attr: 'Contrast' },
  { pos: 'Clean edges without compression artifacts.', neg: 'Heavy blocky JPEG compression artifacts.', cat: 'quality_attribute', attr: 'Compression' },

  // General aesthetic
  { pos: 'A masterpiece photograph.', neg: 'An amateurish snapshot.', cat: 'aesthetic_general', attr: null },
  { pos: 'Award-winning photo on National Geographic.', neg: 'Low-effort accidental snapshot.', cat: 'aesthetic_general', attr: null },
  { pos: 'Visually captivating composition.', neg: 'Visually unappealing composition.', cat: 'aesthetic_general', attr: null },
  { pos: 'Professional photography lighting setup.', neg: 'Harsh unflattering lighting.', cat: 'aesthetic_general', attr: null },
  { pos: 'Breathtaking artistic aesthetic.', neg: 'Ugly and jarring visuals.', cat: 'aesthetic_general', attr: null },

  // General quality
  { pos: 'High resolution crisp 4K detail.', neg: 'Low resolution pixelated image.', cat: 'quality_general', attr: null },
  { pos: 'Clean sensor output without banding.', neg: 'Distorted banding and sensor flaws.', cat: 'quality_general', attr: null },
  { pos: 'Optically perfect sharpness from corner to corner.', neg: 'Severe lens distortion and chromatic aberration.', cat: 'quality_general', attr: null },
  { pos: 'Pristine professional photo quality.', neg: 'Degraded damaged poor quality photo.', cat: 'quality_general', attr: null },
];

// Specialized attribute profiles for curated benchmarks
const PRESET_ATTRIBUTE_PROFILES: Record<string, {
  aesthetic: Record<string, number>;
  quality: Record<string, number>;
  topPosIndices: number[];
  topNegIndices: number[];
}> = {
  'golden-hour-landscape': {
    // Alpine Sunrise Glow: High Color Harmony, Vivid Colors, Good Lighting, Landscape
    aesthetic: {
      'Good Lighting': 94,
      'Color Harmony': 97,
      'Rule of Thirds': 88,
      'Subject Emphasis': 76,
      'Depth of Field': 72,
      'Vivid Colors': 98,
      'Balance & Symmetry': 75,
      'Interesting Content': 92,
      'Motion & Blur': 84,
      'Repetition & Pattern': 70,
    },
    quality: {
      'Blur': 93,
      'Noise': 95,
      'Color-related': 96,
      'Contrast': 91,
      'Compression': 97,
    },
    topPosIndices: [5, 1, 0, 7, 16], // Vivid colors, Harmonious colors, Good lighting, Interesting content, Masterpiece
    topNegIndices: [4, 6],           // Shallow depth of field (landscape prefers deep DoF), Balance & Symmetry
  },
  'cinematic-portrait': {
    // Studio Bokeh Portrait: High Depth of Field, Subject Emphasis, Studio Lighting
    aesthetic: {
      'Good Lighting': 93,
      'Color Harmony': 89,
      'Rule of Thirds': 85,
      'Subject Emphasis': 98,
      'Depth of Field': 97,
      'Vivid Colors': 76,
      'Balance & Symmetry': 82,
      'Interesting Content': 90,
      'Motion & Blur': 88,
      'Repetition & Pattern': 65,
    },
    quality: {
      'Blur': 96,
      'Noise': 92,
      'Color-related': 90,
      'Contrast': 89,
      'Compression': 95,
    },
    topPosIndices: [4, 3, 0, 10, 18], // Shallow DoF, Subject Emphasis, Good lighting, Tack sharp focus, Studio lighting
    topNegIndices: [5, 9],            // Vivid colors (portrait is muted/natural), Repetition
  },
  'urban-architecture': {
    // Geometric Facade: High Balance & Symmetry, Pattern, Contrast, Crisp 4K
    aesthetic: {
      'Good Lighting': 83,
      'Color Harmony': 78,
      'Rule of Thirds': 89,
      'Subject Emphasis': 84,
      'Depth of Field': 65,
      'Vivid Colors': 63,
      'Balance & Symmetry': 98,
      'Interesting Content': 86,
      'Motion & Blur': 90,
      'Repetition & Pattern': 96,
    },
    quality: {
      'Blur': 95,
      'Noise': 91,
      'Color-related': 84,
      'Contrast': 96,
      'Compression': 93,
    },
    topPosIndices: [6, 9, 13, 20, 2], // Symmetry, Pattern, Micro-contrast, 4K detail, Rule of thirds
    topNegIndices: [4, 5],            // Shallow DoF, Vivid colors
  },
  'blurry-night-motion': {
    // Night Motion Blur: Severe motion blur, high noise, poor lighting
    aesthetic: {
      'Good Lighting': 24,
      'Color Harmony': 38,
      'Rule of Thirds': 32,
      'Subject Emphasis': 26,
      'Depth of Field': 28,
      'Vivid Colors': 44,
      'Balance & Symmetry': 30,
      'Interesting Content': 35,
      'Motion & Blur': 18,
      'Repetition & Pattern': 25,
    },
    quality: {
      'Blur': 15,
      'Noise': 22,
      'Color-related': 36,
      'Contrast': 30,
      'Compression': 45,
    },
    topPosIndices: [8],               // Artistic motion blur (only possible saving grace)
    topNegIndices: [10, 11, 0, 13, 15], // Out of focus blurry, Grainy noise, Poor lighting, Low contrast, Amateurish snapshot
  },
};

export function generatePpaAnalysis(seedScore: number = 3.8, sampleId?: string): AnalysisResult {
  const norm = Math.max(0.05, Math.min(0.98, (seedScore - 1.0) / 4.0));
  
  let tier: AnalysisResult['tier'] = 'Good Everyday';
  let tierColor = 'text-amber-400 bg-amber-400/10 border-amber-400/30';
  if (seedScore >= 4.3) {
    tier = 'Masterpiece';
    tierColor = 'text-cyan-400 bg-cyan-400/10 border-cyan-400/30';
  } else if (seedScore >= 3.8) {
    tier = 'High Aesthetic';
    tierColor = 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30';
  } else if (seedScore < 2.5) {
    tier = 'Low / Flawed';
    tierColor = 'text-rose-400 bg-rose-400/10 border-rose-400/30';
  } else if (seedScore < 3.2) {
    tier = 'Sub-optimal';
    tierColor = 'text-orange-400 bg-orange-400/10 border-orange-400/30';
  }

  const profile = sampleId ? PRESET_ATTRIBUTE_PROFILES[sampleId] : null;

  // Create prompt items
  const allPrompts: PromptItem[] = CORE_PROMPT_DATABASE.map((p, idx) => {
    let score: number;
    let rawWeight: number;

    if (profile) {
      if (p.attr && profile.aesthetic[p.attr] !== undefined) {
        score = profile.aesthetic[p.attr] / 100;
      } else if (p.attr && profile.quality[p.attr] !== undefined) {
        score = profile.quality[p.attr] / 100;
      } else if (profile.topPosIndices.includes(idx)) {
        score = Math.min(0.99, norm + 0.12);
      } else if (profile.topNegIndices.includes(idx)) {
        score = Math.max(0.08, norm - 0.28);
      } else {
        score = Math.max(0.05, Math.min(0.98, norm + (Math.sin(idx * 2.1) * 0.08)));
      }
      const isTopPos = profile.topPosIndices.includes(idx);
      const isTopNeg = profile.topNegIndices.includes(idx);
      rawWeight = isTopPos ? 3.5 : isTopNeg ? 2.8 : 0.8 + Math.cos(idx) * 0.4;
    } else {
      const variance = (Math.sin(idx * 1.7) * 0.15);
      score = Math.max(0.05, Math.min(0.99, norm + variance));
      rawWeight = Math.max(0.005, (Math.cos(idx * 2.3) + 1.2) * (score > 0.6 ? 1.4 : 0.8));
    }

    return {
      id: `prompt-${idx}`,
      posPrompt: p.pos,
      negPrompt: p.neg,
      category: p.cat as any,
      attribute: p.attr,
      score: Number(score.toFixed(4)),
      selectionWeight: rawWeight,
      impact: Number((score * rawWeight).toFixed(4)),
    };
  });

  // Normalize selection weights
  const totalWeight = allPrompts.reduce((acc, curr) => acc + curr.selectionWeight, 0);
  allPrompts.forEach(p => {
    p.selectionWeight = Number((p.selectionWeight / totalWeight).toFixed(4));
    p.impact = Number((p.score * p.selectionWeight).toFixed(5));
  });

  // Top positive and negative drivers
  const sortedByImpact = [...allPrompts].sort((a, b) => b.impact - a.impact);
  const sortedByNegImpact = [...allPrompts].sort((a, b) => (a.score * a.selectionWeight) - (b.score * b.selectionWeight));

  const topPositivePrompts = sortedByImpact.slice(0, 6);
  const topNegativePrompts = sortedByNegImpact.filter(p => p.score < 0.65).slice(0, 5);

  // Group by attributes
  const aestheticAttributes = [
    'Good Lighting', 'Color Harmony', 'Rule of Thirds', 'Subject Emphasis',
    'Depth of Field', 'Vivid Colors', 'Balance & Symmetry', 'Interesting Content'
  ];

  const attributeBreakdown = aestheticAttributes.map(name => {
    if (profile && profile.aesthetic[name] !== undefined) {
      return {
        name,
        score: profile.aesthetic[name],
        weight: Math.round(profile.aesthetic[name] * 0.8),
        sampleCount: 1,
      };
    }
    const matching = allPrompts.filter(p => p.attribute === name);
    const avgScore = matching.length ? matching.reduce((a, b) => a + b.score, 0) / matching.length : norm;
    const avgWeight = matching.length ? matching.reduce((a, b) => a + b.selectionWeight, 0) : 0.05;
    return {
      name,
      score: Math.round(avgScore * 100),
      weight: Math.round(avgWeight * 500),
      sampleCount: matching.length,
    };
  });

  const qualityAttributes = ['Blur', 'Noise', 'Color-related', 'Contrast', 'Compression'];
  const qualityBreakdown = qualityAttributes.map(name => {
    if (profile && profile.quality[name] !== undefined) {
      return {
        name,
        score: profile.quality[name],
        weight: Math.round(profile.quality[name] * 0.8),
        sampleCount: 1,
      };
    }
    const matching = allPrompts.filter(p => p.attribute === name);
    const avgScore = matching.length ? matching.reduce((a, b) => a + b.score, 0) / matching.length : norm;
    const avgWeight = matching.length ? matching.reduce((a, b) => a + b.selectionWeight, 0) : 0.05;
    return {
      name,
      score: Math.round(avgScore * 100),
      weight: Math.round(avgWeight * 500),
      sampleCount: matching.length,
    };
  });

  return {
    rawScore: Number(seedScore.toFixed(2)),
    normalizedScore: Number(norm.toFixed(4)),
    percentile: Math.round(norm * 100),
    tier,
    tierColor,
    inferenceTimeMs: Math.round(38 + Math.random() * 24),
    device: 'CUDA:0 (NVIDIA RTX)',
    topPositivePrompts,
    topNegativePrompts,
    allPrompts,
    attributeBreakdown,
    qualityBreakdown,
    categoryWeights: {
      aestheticAttribute: 38,
      qualityAttribute: 24,
      aestheticGeneral: 26,
      qualityGeneral: 12,
    },
  };
}
