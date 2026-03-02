/**
 * Step 4 — Node Ranking.
 *
 * Shows the top-ranked nodes per community (larger dots) and the
 * route start/end endpoints (green/red). Clicking a community in the
 * sidebar highlights it on the map and zooms to its bounds.
 */

import L from 'leaflet';
import type { PipelineStep } from './types.ts';
import type { CommunityDetail } from '../api.ts';
import { fetchCommunities, fetchCommunity, fetchGraphNodes } from '../api.ts';
import { addLayerGroup, removeLayer, getMap } from '../map.ts';
import { communityColour } from '../colours.ts';
import { showLoading } from '../components/loading.ts';

const LAYER_TOP = 'ranking-top';
const LAYER_ENDPOINTS = 'ranking-endpoints';

// Per-step state, cleared in onExit.
const topMarkersByComm = new Map<number, L.CircleMarker[]>();
const endpointsByComm = new Map<number, L.CircleMarker[]>();
const communityBounds = new Map<number, L.LatLngBounds>();
let selectedCommunity: number | null = null;
let listContainer: HTMLElement | null = null;

// ── Highlight / reset ──────────────────────────────────────────────────────

function setMarkersOpacity(
  markersByComm: Map<number, L.CircleMarker[]>,
  active: number | null,
  activeOpacity: number,
  dimOpacity: number,
): void {
  for (const [label, markers] of markersByComm) {
    const isActive = active === null || label === active;
    const fo = isActive ? activeOpacity : dimOpacity;
    for (const m of markers) m.setStyle({ fillOpacity: fo, opacity: fo });
  }
}

function highlightCommunity(label: number): void {
  selectedCommunity = label;
  setMarkersOpacity(topMarkersByComm, label, 0.95, 0.08);
  setMarkersOpacity(endpointsByComm, label, 1, 0.08);

  const bounds = communityBounds.get(label);
  if (bounds?.isValid()) getMap().fitBounds(bounds, { padding: [50, 50] });

  listContainer?.querySelectorAll<HTMLElement>('.community-item').forEach((el) => {
    el.classList.toggle('active', el.dataset['label'] === String(label));
  });
}

function resetHighlight(): void {
  selectedCommunity = null;
  setMarkersOpacity(topMarkersByComm, null, 0.85, 0);
  setMarkersOpacity(endpointsByComm, null, 1, 0);

  listContainer?.querySelectorAll<HTMLElement>('.community-item').forEach((el) => {
    el.classList.remove('active');
  });
}

// ── Sidebar ────────────────────────────────────────────────────────────────

function renderSidebar(container: HTMLElement, details: CommunityDetail[]): void {
  listContainer = container;

  const header = document.createElement('p');
  header.className = 'data-note';
  header.textContent = `${details.length} communities detected. Click a community to zoom and highlight.`;
  container.appendChild(header);

  const list = document.createElement('div');
  list.className = 'community-list';

  for (const comm of details) {
    const color = communityColour(comm.label);
    const item = document.createElement('div');
    item.className = 'community-item';
    item.dataset['label'] = String(comm.label);

    const startText = comm.route_start != null ? String(comm.route_start) : '—';
    const endText = comm.route_end != null ? String(comm.route_end) : '—';

    item.innerHTML = `
      <span class="community-dot" style="background:${color}"></span>
      <div class="community-info">
        <span class="community-name">Community ${comm.label}</span>
        <span class="community-meta">${comm.num_nodes} nodes · ${comm.top_ranked_nodes.length} ranked</span>
        <span class="community-endpoints">↗ ${startText} &rarr; ${endText} ↘</span>
      </div>
    `;

    item.addEventListener('click', () => {
      if (selectedCommunity === comm.label) {
        resetHighlight();
      } else {
        highlightCommunity(comm.label);
      }
    });

    list.appendChild(item);
  }

  container.appendChild(list);
}

// ── Step definition ────────────────────────────────────────────────────────

const step4: PipelineStep = {
  id: 4,
  title: 'Node Ranking',
  narrativeHtml: `
    <p>
      Within each community, nodes are ranked by their proximity to populated
      townlands. The <strong>top-ranked nodes</strong> become candidate bus
      stop locations.
    </p>
    <p>
      The highest-ranked pair in each community is selected as the
      <strong class="label-start">route start</strong> and
      <strong class="label-end">route end</strong>.
    </p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Loading rankings…');

    try {
      const [commList, nodes] = await Promise.all([
        fetchCommunities(),
        fetchGraphNodes(),
      ]);

      const details = await Promise.all(
        commList.communities.map((c) => fetchCommunity(c.label)),
      );

      hideLoading();

      // Build osmid → [lat, lng] lookup from road-network nodes.
      const nodeCoords = new Map<number, [number, number]>();
      for (const feature of nodes.features) {
        const [lng, lat] = feature.geometry.coordinates;
        nodeCoords.set(feature.properties.osmid, [lat, lng]);
      }

      // Build markers per community.
      const allTopMarkers: L.CircleMarker[] = [];
      const allEndpointMarkers: L.CircleMarker[] = [];
      topMarkersByComm.clear();
      endpointsByComm.clear();
      communityBounds.clear();

      for (const comm of details) {
        const color = communityColour(comm.label);
        const topList: L.CircleMarker[] = [];
        const endpointList: L.CircleMarker[] = [];

        for (const osmid of comm.top_ranked_nodes) {
          const coord = nodeCoords.get(osmid);
          if (!coord) continue;

          const m = L.circleMarker(coord, {
            radius: 6,
            color: '#1e293b',
            fillColor: color,
            weight: 1.5,
            fillOpacity: 0.85,
          }).bindTooltip(`Community ${comm.label} — ranked node`);

          topList.push(m);
          allTopMarkers.push(m);

          if (!communityBounds.has(comm.label)) {
            communityBounds.set(comm.label, L.latLngBounds([coord, coord]));
          } else {
            communityBounds.get(comm.label)!.extend(coord);
          }
        }

        if (comm.route_start != null) {
          const coord = nodeCoords.get(comm.route_start);
          if (coord) {
            const m = L.circleMarker(coord, {
              radius: 9,
              color: '#166534',
              fillColor: '#22c55e',
              weight: 2,
              fillOpacity: 1,
            }).bindTooltip(`Route start — community ${comm.label}`);
            endpointList.push(m);
            allEndpointMarkers.push(m);
            communityBounds.get(comm.label)?.extend(coord);
          }
        }

        if (comm.route_end != null) {
          const coord = nodeCoords.get(comm.route_end);
          if (coord) {
            const m = L.circleMarker(coord, {
              radius: 9,
              color: '#991b1b',
              fillColor: '#ef4444',
              weight: 2,
              fillOpacity: 1,
            }).bindTooltip(`Route end — community ${comm.label}`);
            endpointList.push(m);
            allEndpointMarkers.push(m);
            communityBounds.get(comm.label)?.extend(coord);
          }
        }

        topMarkersByComm.set(comm.label, topList);
        endpointsByComm.set(comm.label, endpointList);
      }

      addLayerGroup(LAYER_TOP, allTopMarkers);
      addLayerGroup(LAYER_ENDPOINTS, allEndpointMarkers);

      renderSidebar(sidebar, details);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to load ranking data — is the API running?</p>';
      console.error('[step4]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_TOP);
    removeLayer(LAYER_ENDPOINTS);
    topMarkersByComm.clear();
    endpointsByComm.clear();
    communityBounds.clear();
    selectedCommunity = null;
    listContainer = null;
  },
};

export default step4;
