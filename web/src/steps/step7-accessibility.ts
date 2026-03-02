/**
 * Step 7 — Accessibility Analysis.
 *
 * Each sampled source node is drawn as a circle coloured by how many generated
 * stops are reachable from it (green = high, red = low). The sidebar shows
 * summary stats and a histogram of the reach distribution.
 */

import L from 'leaflet';
import type { PipelineStep } from './types.ts';
import type { AccessibilityTestResult } from '../api.ts';
import { fetchAccessibility, fetchGraphNodes } from '../api.ts';
import { addLayerGroup, removeLayer } from '../map.ts';
import { showLoading } from '../components/loading.ts';
import { createStatsTable } from '../components/stats-table.ts';
import { createBarChart } from '../components/bar-chart.ts';

const LAYER_REACH = 'accessibility-markers';
const BINS = 8;

function reachColour(rate: number): string {
  if (rate >= 0.8) return '#16a34a';
  if (rate >= 0.6) return '#65a30d';
  if (rate >= 0.4) return '#ca8a04';
  if (rate >= 0.2) return '#ea580c';
  return '#dc2626';
}

function buildHistogram(
  results: AccessibilityTestResult[],
  destTotal: number,
): { label: string; value: number; color: string }[] {
  const counts = Array<number>(BINS).fill(0);
  for (const r of results) {
    const rate = destTotal > 0 ? r.destinations_reached / destTotal : 0;
    const bin = Math.min(Math.floor(rate * BINS), BINS - 1);
    counts[bin]++;
  }
  return counts.map((count, i) => {
    const rate = (i + 0.5) / BINS;
    return {
      label: `${Math.round((i / BINS) * 100)}%`,
      value: count,
      color: reachColour(rate),
    };
  });
}

const step7: PipelineStep = {
  id: 7,
  title: 'Accessibility',
  narrativeHtml: `
    <p>
      Each node in the sample is tested: how many generated bus stops are
      reachable via the road network?
    </p>
    <p>
      Nodes are coloured by their reach rate —
      <span style="color:#16a34a;font-weight:600">green</span> is high
      accessibility,
      <span style="color:#dc2626;font-weight:600">red</span> is low.
    </p>
    <p class="data-note">
      First load may take 10–30 seconds while the server runs accessibility tests.
    </p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Running accessibility tests… (may take 30 s)');

    try {
      const [summary, nodes] = await Promise.all([
        fetchAccessibility(),
        fetchGraphNodes(),
      ]);

      hideLoading();

      // Build osmid → [lat, lng] lookup.
      const nodeCoords = new Map<number, [number, number]>();
      for (const f of nodes.features) {
        const [lng, lat] = f.geometry.coordinates;
        nodeCoords.set(f.properties.osmid, [lat, lng]);
      }

      // Place markers coloured by reach rate.
      const destTotal =
        summary.results.length > 0 ? (summary.results[0]?.destinations_total ?? 0) : 0;

      const markers = summary.results
        .map((r) => {
          const coord = nodeCoords.get(r.source_node);
          if (!coord) return null;
          const rate = destTotal > 0 ? r.destinations_reached / destTotal : 0;
          const colour = reachColour(rate);
          return L.circleMarker(coord, {
            radius: 6,
            color: '#00000033',
            fillColor: colour,
            weight: 0,
            fillOpacity: 0.8,
          }).bindTooltip(
            `Reached ${r.destinations_reached} / ${destTotal} stops (${Math.round(rate * 100)}%)`,
          );
        })
        .filter((m): m is L.CircleMarker => m !== null);

      addLayerGroup(LAYER_REACH, markers);

      // ── Sidebar ──────────────────────────────────────────────────────────

      const pct = destTotal > 0
        ? ((summary.avg_destinations_reached / destTotal) * 100).toFixed(1)
        : '—';

      const table = createStatsTable([
        { label: 'Tests run', value: String(summary.total_tests) },
        { label: 'Avg stops reached', value: `${summary.avg_destinations_reached.toFixed(1)} of ${destTotal}` },
        { label: 'Avg reach rate', value: `${pct}%`, highlight: true },
        { label: 'Avg path length', value: `${summary.avg_path_length.toFixed(2)} km` },
      ]);

      const chartTitle = document.createElement('p');
      chartTitle.className = 'chart-title';
      chartTitle.textContent = 'Distribution of stops reached';

      const histData = buildHistogram(summary.results, destTotal);
      const chart = createBarChart(histData, {
        height: 130,
        formatValue: String,
      });

      sidebar.appendChild(table);
      sidebar.appendChild(chartTitle);
      sidebar.appendChild(chart);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to run accessibility tests — is the API running?</p>';
      console.error('[step7]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_REACH);
  },
};

export default step7;
