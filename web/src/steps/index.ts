/**
 * Pipeline step definitions.
 *
 * Steps are registered here and added incrementally across implementation phases.
 * Phase 2: steps 0–1 (Introduction, Road Network)
 */

export type { PipelineStep } from './types.ts';

import type { PipelineStep } from './types.ts';
import step0 from './step0-intro.ts';
import step1 from './step1-network.ts';

export const steps: PipelineStep[] = [step0, step1];
