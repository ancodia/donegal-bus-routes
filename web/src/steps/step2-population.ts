/**
 * Step 2 placeholder — implemented in Phase 3.
 */

import type { PipelineStep } from './types.ts';

const step2: PipelineStep = {
  id: 2,
  title: 'Population Data',
  narrativeHtml: `
    <p>
      Edges are weighted by the population of nearby townlands, making
      densely populated corridors more attractive for bus routing.
    </p>
    <p class="data-note">Full population visualisation coming in Phase 3.</p>
  `,
  async onEnter(_map, _sidebar) {},
  onExit(_map) {},
};

export default step2;
