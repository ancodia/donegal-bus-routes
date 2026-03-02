/**
 * Step 5 — Generated Routes.
 *
 * Renders all community routes as coloured LineStrings, generated bus stop
 * markers, and connecting routes as dashed grey polylines.
 *
 * Clicking a route card in the sidebar isolates that route on the map.
 */

import L from 'leaflet';
import type { PipelineStep } from './types.ts';
import type { RouteFeatureProperties } from '../api.ts';
import {
  fetchRoutesGeoJSON,
  fetchGeneratedStops,
  fetchRoutes,
  fetchConnectionRoute,
} from '../api.ts';
import { addGeoJSONLayer, addLayerGroup, removeLayer, getMap } from '../map.ts';
import { communityColour } from '../colours.ts';
import { showLoading } from '../components/loading.ts';
import { createRouteCard } from '../components/route-card.ts';

const LAYER_ROUTES = 'routes-lines';
const LAYER_STOPS = 'routes-stops';
const LAYER_CONNECTIONS = 'routes-connections';

const CONNECTION_LABELS = ['a', 'b', 'c', 'd'] as const;

// Per-step state.
type RouteGeoLayer = L.GeoJSON<RouteFeatureProperties>;
let routeLayer: RouteGeoLayer | null = null;
let stopLayer: L.GeoJSON | null = null;
let selectedCommunity: number | null = null;
let sidebarRef: HTMLElement | null = null;

// ── Selection logic ────────────────────────────────────────────────────────

function selectRoute(label: number): void {
  selectedCommunity = label;

  routeLayer?.eachLayer((l) => {
    const pl = l as L.Polyline & { feature?: GeoJSON.Feature<GeoJSON.Geometry, RouteFeatureProperties> };
    const isSelected = pl.feature?.properties?.community === label;
    pl.setStyle({ opacity: isSelected ? 1 : 0.12, weight: isSelected ? 4 : 1.5 });
    if (isSelected) pl.bringToFront();
  });

  stopLayer?.eachLayer((l) => {
    const m = l as L.CircleMarker & { feature?: GeoJSON.Feature };
    const community = m.feature?.properties?.['community'] as number | undefined;
    const active = community === label;
    m.setStyle({ fillOpacity: active ? 0.95 : 0.08, opacity: active ? 1 : 0.15 });
  });

  // Zoom to selected route.
  routeLayer?.eachLayer((l) => {
    const pl = l as L.Polyline & { feature?: GeoJSON.Feature<GeoJSON.Geometry, RouteFeatureProperties> };
    if (pl.feature?.properties?.community === label) {
      const bounds = pl.getBounds();
      if (bounds.isValid()) getMap().fitBounds(bounds, { padding: [60, 60] });
    }
  });

  sidebarRef?.querySelectorAll<HTMLElement>('.route-card').forEach((card) => {
    card.classList.toggle('active', card.dataset['community'] === String(label));
  });
}

function resetRoutes(): void {
  selectedCommunity = null;

  routeLayer?.setStyle((feature) => ({
    color: communityColour((feature?.properties as RouteFeatureProperties | undefined)?.community ?? 0),
    opacity: 0.85,
    weight: 2.5,
  }));

  stopLayer?.eachLayer((l) => {
    (l as L.CircleMarker).setStyle({ fillOpacity: 0.9, opacity: 1 });
  });

  sidebarRef?.querySelectorAll<HTMLElement>('.route-card').forEach((card) => {
    card.classList.remove('active');
  });
}

// ── Sidebar ────────────────────────────────────────────────────────────────

function renderSidebar(
  container: HTMLElement,
  routeFeatures: GeoJSON.Feature<GeoJSON.Geometry, RouteFeatureProperties>[],
  stopCount: number,
  connCount: number,
): void {
  sidebarRef = container;

  const headerRow = document.createElement('div');
  headerRow.className = 'routes-header';
  headerRow.innerHTML = `
    <span class="data-note">${routeFeatures.length} routes &middot; ${stopCount} stops${connCount > 0 ? ` &middot; ${connCount} connectors` : ''}</span>
    <button class="btn-reset" id="routes-reset">Show all</button>
  `;
  container.appendChild(headerRow);

  const list = document.createElement('div');
  list.className = 'route-cards';

  for (const feature of routeFeatures) {
    const props = feature.properties;
    const color = communityColour(props.community);
    const card = createRouteCard(
      { community: props.community, numStops: props.num_stops, color },
      () => {
        if (selectedCommunity === props.community) {
          resetRoutes();
        } else {
          selectRoute(props.community);
        }
      },
    );
    list.appendChild(card);
  }

  container.appendChild(list);

  container.querySelector<HTMLButtonElement>('#routes-reset')?.addEventListener('click', resetRoutes);
}

// ── Step definition ────────────────────────────────────────────────────────

const step5: PipelineStep = {
  id: 5,
  title: 'Generated Routes',
  narrativeHtml: `
    <p>
      For each community, a bus route is generated connecting the
      top-ranked nodes. Separate <strong>connecting routes</strong> link
      adjacent communities.
    </p>
    <p>Click a route card to isolate it on the map.</p>
  `,

  async onEnter(_map, sidebar) {
    const hideLoading = showLoading(sidebar, 'Loading routes…');

    try {
      const [routeGeoJSON, stopsGeoJSON, routeList] = await Promise.all([
        fetchRoutesGeoJSON(),
        fetchGeneratedStops(),
        fetchRoutes(),
      ]);

      // Fetch connection routes — some labels may not exist (404).
      const connResults = await Promise.allSettled(
        CONNECTION_LABELS.map((lbl) => fetchConnectionRoute(lbl)),
      );

      hideLoading();

      // ── Route lines ──
      routeLayer = addGeoJSONLayer(LAYER_ROUTES, routeGeoJSON, {
        style: (feature) => ({
          color: communityColour(
            (feature?.properties as RouteFeatureProperties | undefined)?.community ?? 0,
          ),
          weight: 2.5,
          opacity: 0.85,
        }),
      }) as RouteGeoLayer;

      // ── Connecting routes (dashed) ──
      const connPolylines: L.Polyline[] = [];
      for (const result of connResults) {
        if (result.status !== 'fulfilled' || result.value.stops.length < 2) continue;
        const latlngs = result.value.stops.map((s) => L.latLng(s.y, s.x));
        connPolylines.push(
          L.polyline(latlngs, {
            color: '#475569',
            weight: 2,
            opacity: 0.75,
            dashArray: '8 5',
          }).bindTooltip('Connecting route'),
        );
      }
      addLayerGroup(LAYER_CONNECTIONS, connPolylines);

      // ── Stop markers ──
      stopLayer = addGeoJSONLayer(LAYER_STOPS, stopsGeoJSON, {
        pointToLayer: (feature, latlng) => {
          const community = feature.properties?.['community'] as number | undefined;
          const color = communityColour(community ?? 0);
          return L.circleMarker(latlng, {
            radius: 5,
            color,
            fillColor: '#ffffff',
            weight: 2,
            fillOpacity: 0.9,
          });
        },
      });

      // Count community-route stops for the header.
      const stopCount = routeList.routes.reduce((sum, r) => sum + r.stops.length, 0);

      renderSidebar(sidebar, routeGeoJSON.features, stopCount, connPolylines.length);
    } catch (err) {
      hideLoading();
      sidebar.innerHTML =
        '<p class="error-msg">Failed to load routes — is the API running?</p>';
      console.error('[step5]', err);
    }
  },

  onExit(_map) {
    removeLayer(LAYER_ROUTES);
    removeLayer(LAYER_STOPS);
    removeLayer(LAYER_CONNECTIONS);
    routeLayer = null;
    stopLayer = null;
    selectedCommunity = null;
    sidebarRef = null;
  },
};

export default step5;
