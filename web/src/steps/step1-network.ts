import type { PipelineStep } from './types.ts';
import type { GraphSummary } from '../api.ts';
import { fetchGraphSummary, fetchGraphEdges, fetchGraphNodes } from '../api.ts';
import { removeLayer, fitBoundsToLayer } from '../map.ts';
import { addNodeLayer, addEdgeLayer } from '../layers.ts';
import { showLoading } from '../components/loading.ts';

const LAYER_EDGES = 'network-edges';
const LAYER_NODES = 'network-nodes';

function renderStats(container: HTMLElement, s: GraphSummary): void {
  container.innerHTML = `
    <div class="stat-grid">
      <div class="stat">
        <span class="stat-value">${s.num_nodes.toLocaleString()}</span>
        <span class="stat-label">Nodes</span>
      </div>
      <div class="stat">
        <span class="stat-value">${s.num_edges.toLocaleString()}</span>
        <span class="stat-label">Edges</span>
      </div>
      <div class="stat">
        <span class="stat-value">${s.num_communities}</span>
        <span class="stat-label">Communities</span>
      </div>
      <div class="stat">
        <span class="stat-value">${s.is_connected ? 'Yes' : 'No'}</span>
        <span class="stat-label">Connected</span>
      </div>
    </div>
    <p class="data-note">
      Blue circles are road junctions (nodes). Grey lines are road segments
      (edges). Each edge carries a population weight used in later steps.
    </p>
  `;
}

const step1: PipelineStep = {
  id: 1,
  title: 'Road Network',
  narrativeHtml: `
    <p>
      The pipeline starts with an <strong>OpenStreetMap road network</strong>
      for County Donegal, downloaded via OSMnx and stored as a GraphML file.
    </p>
    <p>
      Each road segment is weighted by the population of nearby townlands,
      making densely populated corridors more attractive for bus routing.
    </p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Loading road network…');
    try {
      const [summary, edges, nodes] = await Promise.all([
        fetchGraphSummary(),
        fetchGraphEdges(),
        fetchGraphNodes(),
      ]);

      hideLoading();
      renderStats(sidebar, summary);

      // Edges first so nodes render on top.
      addEdgeLayer(LAYER_EDGES, edges);
      addNodeLayer(LAYER_NODES, nodes);
      fitBoundsToLayer(LAYER_EDGES);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to load road network — is the API running?</p>';
      console.error('[step1]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_EDGES);
    removeLayer(LAYER_NODES);
  },
};

export default step1;
