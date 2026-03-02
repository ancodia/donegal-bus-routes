/**
 * Pipeline step definitions.
 *
 * Phase 2: steps 0–1  (Introduction, Road Network)
 * Phase 4: steps 2–5  (Population placeholder, Communities placeholder, Ranking, Routes)
 * Phase 5: steps 6–8  (Existing Routes, Accessibility, Cost Comparison)
 */

export type { PipelineStep } from './types.ts';

import type { PipelineStep } from './types.ts';
import step0 from './step0-intro.ts';
import step1 from './step1-network.ts';
import step2 from './step2-population.ts';
import step3 from './step3-communities.ts';
import step4 from './step4-ranking.ts';
import step5 from './step5-routes.ts';
import step6 from './step6-actual.ts';
import step7 from './step7-accessibility.ts';
import step8 from './step8-cost.ts';

export const steps: PipelineStep[] = [step0, step1, step2, step3, step4, step5, step6, step7, step8];
