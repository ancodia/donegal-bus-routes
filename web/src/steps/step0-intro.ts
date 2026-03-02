import type { PipelineStep } from './types.ts';

const step0: PipelineStep = {
  id: 0,
  title: 'Introduction',
  narrativeHtml: `
    <p>
      This dashboard walks through a pipeline for generating optimised rural bus
      routes in County Donegal using community detection and
      population-weighted graph analysis.
    </p>
    <h3>Study Region</h3>
    <p>
      County Donegal is one of the most rural counties in Ireland — over 4,800&nbsp;km²
      with a dispersed population of ~160,000. Public transport coverage is thin,
      serving primarily urban centres along the coast.
    </p>
    <h3>Pipeline Aims</h3>
    <ul>
      <li>Model the road network as a population-weighted graph</li>
      <li>Detect natural geographic communities (Louvain method)</li>
      <li>Rank nodes by population importance within each community</li>
      <li>Generate candidate bus routes connecting top-ranked nodes</li>
      <li>Compare coverage against existing LocalLink services</li>
    </ul>
    <p class="data-note">
      Navigate the steps above to walk through each stage of the pipeline.
    </p>
  `,

  async onEnter(_map, _sidebar) {
    // Static step — the base map centred on Donegal is sufficient.
  },

  onExit(_map) {
    // Nothing to clean up.
  },
};

export default step0;
