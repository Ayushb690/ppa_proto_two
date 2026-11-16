export interface PromptItem {
  id: string;
  posPrompt: string;
  negPrompt: string;
  category: 'aesthetic_attribute' | 'quality_attribute' | 'aesthetic_general' | 'quality_general';
  attribute: string | null;
  selectionWeight: number; // p_phi (0 - 1)
  score: number;           // s_bar (0 - 1)
  impact: number;          // weight * score
}

export interface AttributeScore {
  name: string;
  score: number;   // 0 - 100
  weight: number;  // 0 - 100
  sampleCount: number;
}

export interface AnalysisResult {
  rawScore: number;         // 1.0 - 5.0 scale (PARA)
  normalizedScore: number;  // 0.0 - 1.0 scale
  percentile: number;       // 0 - 100%
  tier: 'Masterpiece' | 'High Aesthetic' | 'Good Everyday' | 'Sub-optimal' | 'Low / Flawed';
  tierColor: string;
  inferenceTimeMs: number;
  device: string;
  topPositivePrompts: PromptItem[];
  topNegativePrompts: PromptItem[];
  allPrompts: PromptItem[];
  attributeBreakdown: AttributeScore[];
  qualityBreakdown: AttributeScore[];
  categoryWeights: {
    aestheticAttribute: number;
    qualityAttribute: number;
    aestheticGeneral: number;
    qualityGeneral: number;
  };
}

export interface SampleImage {
  id: string;
  title: string;
  category: string;
  description: string;
  url: string;
  expectedScore: number;
}
