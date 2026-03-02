/**
 * Step 3 placeholder — implemented in Phase 3.
 */

import type { PipelineStep } from './types.ts';

const step3: PipelineStep = {
  id: 3,
  title: 'Community Detection',
  narrativeHtml: `
    <p>
      The Louvain algorithm detects 14 natural geographic communities in the
      weighted road graph. Each community becomes the basis for a bus route.
    </p>
    <p class="data-note">Full community colouring coming in Phase 3.</p>
  `,
  async onEnter(_map, _sidebar) {},
  onExit(_map) {},
};

export default step3;
