/**
 * Step 9 — Nearest Stop Finder.
 *
 * Switches the map cursor to a crosshair. On click, queries the API for the
 * nearest generated stop, draws a connector line, and opens a popup.
 * The sidebar shows the last query result.
 */

import L from 'leaflet';
import type { PipelineStep } from './types.ts';
import { fetchNearestStop } from '../api.ts';
import { addLayerGroup, removeLayer, getMap } from '../map.ts';
import { showLoading } from '../components/loading.ts';
import { createNearestStopPopup } from '../components/popup.ts';

const LAYER_CLICK = 'finder-click';
const LAYER_NEAREST = 'finder-nearest';
const LAYER_LINE = 'finder-line';

// Donegal is ~55°N. Distance from API is hypot(Δlng_deg, Δlat_deg), which mixes
// scales (~111 km/° lat vs ~64 km/° lng). Using 90 km/° as a reasonable midpoint.
const KM_PER_DEG = 90;

type ClickHandler = (e: L.LeafletMouseEvent) => void;

let _clickHandler: ClickHandler | null = null;
let _sidebarRef: HTMLElement | null = null;

// ── Sidebar states ─────────────────────────────────────────────────────────

function showInstructions(container: HTMLElement): void {
  container.innerHTML = `
    <div class="finder-instructions">
      <p>Click anywhere on the map to find the nearest generated bus stop.</p>
      <p>A line will be drawn from your click point to the stop.</p>
      <p class="data-note">Distances are approximate (±15%).</p>
    </div>
  `;
}

function showResult(
  container: HTMLElement,
  result: { osmid: number; community: number; distanceKm: number; lat: number; lng: number },
): void {
  const wLng = Math.abs(result.lng).toFixed(4);
  const nLat = result.lat.toFixed(4);
  container.innerHTML = `
    <div class="finder-result">
      <p class="finder-result-header">Nearest stop found</p>
      <div class="stats-table">
        <div class="stats-row">
          <span class="stats-label">Community</span>
          <span class="stats-value">${result.community}</span>
        </div>
        <div class="stats-row">
          <span class="stats-label">OSM ID</span>
          <span class="stats-value">${result.osmid}</span>
        </div>
        <div class="stats-row highlight">
          <span class="stats-label">Distance</span>
          <span class="stats-value">~${result.distanceKm.toFixed(2)} km</span>
        </div>
        <div class="stats-row">
          <span class="stats-label">Click point</span>
          <span class="stats-value">${nLat}°N, ${wLng}°W</span>
        </div>
      </div>
      <p class="data-note">Click elsewhere to search again.</p>
    </div>
  `;
}

// ── Step definition ────────────────────────────────────────────────────────

const step9: PipelineStep = {
  id: 9,
  title: 'Nearest Stop Finder',
  narrativeHtml: `
    <p>
      Click anywhere on the map to find the <strong>nearest generated bus
      stop</strong>. A line connects your click point to the stop, and the
      distance is shown below.
    </p>
  `,

  async onEnter(map, sidebar) {
    _sidebarRef = sidebar;
    showInstructions(sidebar);

    getMap().getContainer().style.cursor = 'crosshair';

    const handler: ClickHandler = async (e) => {
      if (!_sidebarRef) return;

      const { lat, lng } = e.latlng;

      // Remove previous result layers.
      removeLayer(LAYER_LINE);
      removeLayer(LAYER_CLICK);
      removeLayer(LAYER_NEAREST);

      // Place a marker at the click point immediately.
      addLayerGroup(LAYER_CLICK, [
        L.circleMarker([lat, lng], {
          radius: 5,
          color: '#475569',
          fillColor: '#94a3b8',
          weight: 1.5,
          fillOpacity: 0.85,
        }),
      ]);

      const hideLoading = showLoading(_sidebarRef, 'Finding nearest stop…');

      try {
        const result = await fetchNearestStop(lat, lng);
        hideLoading();

        if (!result.features.length) {
          _sidebarRef.innerHTML = '<p class="error-msg">No stops found near that location.</p>';
          return;
        }

        const feature = result.features[0]!;
        const [stopLng, stopLat] = feature.geometry.coordinates;
        const props = feature.properties;
        const osmid = props['osmid'] as number;
        const community = props['community'] as number;
        const distanceDeg = props['distance_deg'] as number;
        const distanceKm = distanceDeg * KM_PER_DEG;

        // Connector line (dashed).
        addLayerGroup(LAYER_LINE, [
          L.polyline([[lat, lng], [stopLat, stopLng]], {
            color: '#64748b',
            weight: 2,
            dashArray: '6 4',
            opacity: 0.85,
          }),
        ]);

        // Nearest stop marker with popup.
        const nearestMarker = L.circleMarker([stopLat, stopLng], {
          radius: 10,
          color: '#166534',
          fillColor: '#22c55e',
          weight: 2,
          fillOpacity: 1,
        }).bindPopup(createNearestStopPopup({ osmid, community, distanceKm }), {
          maxWidth: 220,
        });

        addLayerGroup(LAYER_NEAREST, [nearestMarker]);
        nearestMarker.openPopup();

        showResult(_sidebarRef, { osmid, community, distanceKm, lat, lng });
      } catch (err) {
        hideLoading();
        _sidebarRef.innerHTML =
          '<p class="error-msg">Failed to query nearest stop — is the API running?</p>';
        console.error('[step9]', err);
      }
    };

    _clickHandler = handler;
    map.on('click', handler);
  },

  onExit(map) {
    if (_clickHandler) {
      map.off('click', _clickHandler);
      _clickHandler = null;
    }
    removeLayer(LAYER_LINE);
    removeLayer(LAYER_CLICK);
    removeLayer(LAYER_NEAREST);
    getMap().getContainer().style.cursor = '';
    _sidebarRef = null;
  },
};

export default step9;
