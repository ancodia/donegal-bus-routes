/**
 * Pipeline step definitions.
 *
 * Each step corresponds to a stage in the dissertation pipeline.
 * Steps are added incrementally across implementation phases.
 * Phase 2 will add steps 0–1; Phase 3 will add steps 2–3, etc.
 */

import type L from 'leaflet';

export interface PipelineStep {
  id: number;
  title: string;
  narrativeHtml: string;
  onEnter: (map: L.Map, sidebar: HTMLElement) => Promise<void>;
  onExit: (map: L.Map) => void;
}

export const steps: PipelineStep[] = [];
