/**
 * Step 6 — Existing Routes.
 *
 * Overlays actual LocalLink stops (orange) with generated stops (blue) for
 * a direct visual coverage comparison.
 */

import type { PipelineStep } from './types.ts';
import { fetchActualStops, fetchGeneratedStops } from '../api.ts';
import { addGeoJSONLayer, removeLayer } from '../map.ts';
import { showLoading } from '../components/loading.ts';
import { createStatsTable } from '../components/stats-table.ts';
import L from 'leaflet';

const LAYER_GENERATED = 'actual-step-generated';
const LAYER_ACTUAL = 'actual-step-actual';

const step6: PipelineStep = {
  id: 6,
  title: 'Existing Routes',
  narrativeHtml: `
    <p>
      The existing <strong>LocalLink</strong> service in Donegal is overlaid
      against the generated route network for direct comparison.
    </p>
    <p>
      <span class="legend-swatch" style="background:#2563eb"></span>
      <strong>Blue circles</strong> — generated stops.<br>
      <span class="legend-swatch" style="background:#f97316"></span>
      <strong>Orange circles</strong> — actual LocalLink stops.
    </p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Loading stop data…');

    try {
      const [generated, actual] = await Promise.all([
        fetchGeneratedStops(),
        fetchActualStops(),
      ]);

      hideLoading();

      // Generated stops — blue, smaller
      addGeoJSONLayer(LAYER_GENERATED, generated, {
        pointToLayer: (_f, latlng) =>
          L.circleMarker(latlng, {
            radius: 5,
            color: '#1d4ed8',
            fillColor: '#2563eb',
            weight: 1,
            fillOpacity: 0.75,
          }),
      });

      // Actual stops — orange, larger, distinctive
      addGeoJSONLayer(LAYER_ACTUAL, actual, {
        pointToLayer: (_f, latlng) =>
          L.circleMarker(latlng, {
            radius: 7,
            color: '#c2410c',
            fillColor: '#f97316',
            weight: 1.5,
            fillOpacity: 0.9,
          }).bindTooltip('LocalLink stop'),
      });

      const genCount = generated.features.length;
      const actCount = actual.features.length;

      const table = createStatsTable([
        { label: 'Generated stops', value: String(genCount) },
        { label: 'Actual LocalLink stops', value: String(actCount) },
        { label: 'Coverage ratio', value: `${(genCount / actCount).toFixed(1)}×` },
      ]);

      const note = document.createElement('p');
      note.className = 'data-note';
      note.textContent =
        'The generated network serves more locations, reaching dispersed communities ' +
        'that the existing service does not cover.';

      sidebar.appendChild(table);
      sidebar.appendChild(note);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to load stop data — is the API running?</p>';
      console.error('[step6]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_GENERATED);
    removeLayer(LAYER_ACTUAL);
  },
};

export default step6;
