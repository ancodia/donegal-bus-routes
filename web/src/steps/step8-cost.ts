/**
 * Step 8 — Cost Comparison.
 *
 * Compares estimated fuel costs of generated routes vs actual LocalLink routes.
 * The map shows generated routes as context; all the analysis is in the sidebar.
 */

import type { PipelineStep } from './types.ts';
import type { RouteCost, RouteFeatureProperties } from '../api.ts';
import { fetchCost, fetchRoutesGeoJSON } from '../api.ts';
import { addGeoJSONLayer, removeLayer } from '../map.ts';
import { communityColour } from '../colours.ts';
import { showLoading } from '../components/loading.ts';
import { createStatsTable } from '../components/stats-table.ts';
import { createBarChart } from '../components/bar-chart.ts';

const LAYER_ROUTES = 'cost-routes';

function fmt(eur: number): string {
  return `€${eur.toLocaleString('en-IE', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

function diffLabel(gen: number, actual: number): string {
  const pct = ((gen - actual) / actual) * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}% vs actual`;
}

function perRouteSection(
  routes: RouteCost[],
  heading: string,
  labelFn: (r: RouteCost, i: number) => string,
): HTMLElement {
  const wrapper = document.createElement('div');

  const h = document.createElement('p');
  h.className = 'chart-title';
  h.textContent = heading;
  wrapper.appendChild(h);

  const table = document.createElement('div');
  table.className = 'stats-table compact';

  routes.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = 'stats-row';
    row.innerHTML = `
      <span class="stats-label">${labelFn(r, i)}</span>
      <span class="stats-value">${fmt(r.estimated_cost_eur)} · ${r.distance_km.toFixed(1)} km</span>
    `;
    table.appendChild(row);
  });

  wrapper.appendChild(table);
  return wrapper;
}

const step8: PipelineStep = {
  id: 8,
  title: 'Cost Comparison',
  narrativeHtml: `
    <p>
      Estimated fuel costs are compared between the generated routes and the
      existing LocalLink service using a standard fuel consumption model.
    </p>
    <p class="data-note">
      Cost model: 10 L/100 km at €1.70/L. Costs cover a single complete circuit
      of each route.
    </p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Computing cost comparison…');

    try {
      const [costData, routeGeoJSON] = await Promise.all([
        fetchCost(),
        fetchRoutesGeoJSON(),
      ]);

      hideLoading();

      // Show routes on map as context.
      addGeoJSONLayer(LAYER_ROUTES, routeGeoJSON, {
        style: (feature) => ({
          color: communityColour(
            (feature?.properties as RouteFeatureProperties | undefined)?.community ?? 0,
          ),
          weight: 2.5,
          opacity: 0.75,
        }),
      });

      // ── Summary comparison ────────────────────────────────────────────────

      const gen = costData.total_generated_cost;
      const act = costData.total_actual_cost;

      const summaryTable = createStatsTable([
        { label: 'Generated routes total', value: fmt(gen) },
        { label: 'Actual routes total', value: fmt(act) },
        { label: 'Difference', value: diffLabel(gen, act), highlight: true },
      ]);

      // ── Bar chart ─────────────────────────────────────────────────────────

      const chartTitle = document.createElement('p');
      chartTitle.className = 'chart-title';
      chartTitle.textContent = 'Total route cost comparison';

      const chart = createBarChart(
        [
          { label: 'Generated', value: Math.round(gen), color: '#2563eb' },
          { label: 'Actual', value: Math.round(act), color: '#64748b' },
        ],
        { height: 130, formatValue: fmt },
      );

      // ── Per-route breakdown ───────────────────────────────────────────────

      const genSection = perRouteSection(
        costData.generated_routes,
        'Generated routes',
        (r) => `Route ${r.route_community}`,
      );

      const actSection = perRouteSection(
        costData.actual_routes,
        'Actual LocalLink routes',
        (_r, i) => `LocalLink ${i + 1}`,
      );

      sidebar.appendChild(summaryTable);
      sidebar.appendChild(chartTitle);
      sidebar.appendChild(chart);
      sidebar.appendChild(genSection);
      sidebar.appendChild(actSection);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to load cost data — is the API running?</p>';
      console.error('[step8]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_ROUTES);
  },
};

export default step8;
