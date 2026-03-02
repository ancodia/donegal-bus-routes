/**
 * Pipeline step definitions.
 *
 * Steps are registered here and added incrementally across implementation phases.
 * Phase 2: steps 0–1  (Introduction, Road Network)
 * Phase 4: steps 2–5  (Population placeholder, Communities placeholder, Ranking, Routes)
 */

export type { PipelineStep } from './types.ts';

import type { PipelineStep } from './types.ts';
import step0 from './step0-intro.ts';
import step1 from './step1-network.ts';
import step2 from './step2-population.ts';
import step3 from './step3-communities.ts';
import step4 from './step4-ranking.ts';
import step5 from './step5-routes.ts';

export const steps: PipelineStep[] = [step0, step1, step2, step3, step4, step5];
