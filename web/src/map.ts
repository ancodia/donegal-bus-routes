/**
 * Leaflet map initialisation and layer management.
 */

import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Donegal centre — ~54.95°N, 7.73°W
const DONEGAL_CENTRE: L.LatLngExpression = [54.95, -7.73];
const DONEGAL_ZOOM = 9;

// CartoDB Positron — clean, light base map that keeps data layers prominent.
const TILE_URL = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ' +
  '&copy; <a href="https://carto.com/attributions">CARTO</a>';

const MASK_STYLE: L.PathOptions = {
  color: 'transparent',
  weight: 0,
  fillColor: '#1a3557',
  fillOpacity: 0.15,
  interactive: false,
};

const BOUNDARY_STYLE: L.PathOptions = {
  color: '#1a3557',
  weight: 2,
  opacity: 0.45,
  fillColor: '#1a3557',
  fillOpacity: 0.04,
  dashArray: '6 4',
  interactive: false,
};

let _map: L.Map | null = null;
const _layers = new Map<string, L.Layer>();

export function initMap(containerId: string): L.Map {
  _map = L.map(containerId, { preferCanvas: true }).setView(DONEGAL_CENTRE, DONEGAL_ZOOM);

  L.tileLayer(TILE_URL, {
    attribution: ATTRIBUTION,
    subdomains: 'abcd',
    maxZoom: 20,
  }).addTo(_map);

  initMask(_map).catch((err: unknown) => {
    console.warn('[map] mask overlay unavailable:', err);
  });
  initBoundary(_map).catch((err: unknown) => {
    console.warn('[map] boundary overlay unavailable:', err);
  });

  return _map;
}

async function initMask(map: L.Map): Promise<void> {
  const res = await fetch('/donegal-mask.geojson');
  if (!res.ok) throw new Error(`fetch /donegal-mask.geojson → HTTP ${res.status}`);
  const geojson = (await res.json()) as GeoJSON.GeoJsonObject;
  L.geoJSON(geojson, { style: () => MASK_STYLE, interactive: false })
    .addTo(map)
    .bringToBack();
}

async function initBoundary(map: L.Map): Promise<void> {
  const res = await fetch('/donegal-boundary.geojson');
  if (!res.ok) throw new Error(`fetch /donegal-boundary.geojson → HTTP ${res.status}`);
  const geojson = (await res.json()) as GeoJSON.GeoJsonObject;
  L.geoJSON(geojson, { style: () => BOUNDARY_STYLE, interactive: false }).addTo(map);
}

export function getMap(): L.Map {
  if (!_map) throw new Error('Map not initialised — call initMap() first');
  return _map;
}

export function addGeoJSONLayer(
  id: string,
  geojson: GeoJSON.GeoJsonObject,
  options?: L.GeoJSONOptions,
): L.GeoJSON {
  removeLayer(id);
  const layer = L.geoJSON(geojson, options).addTo(getMap());
  _layers.set(id, layer);
  return layer;
}

export function removeLayer(id: string): void {
  const layer = _layers.get(id);
  if (layer) {
    getMap().removeLayer(layer);
    _layers.delete(id);
  }
}

export function clearLayers(): void {
  const map = getMap();
  for (const layer of _layers.values()) {
    map.removeLayer(layer);
  }
  _layers.clear();
}

export function addLayerGroup(id: string, layers: L.Layer[]): L.LayerGroup {
  removeLayer(id);
  const group = L.layerGroup(layers).addTo(getMap());
  _layers.set(id, group);
  return group;
}

export function fitBoundsToLayer(id: string): void {
  const layer = _layers.get(id);
  if (layer && layer instanceof L.GeoJSON) {
    const bounds = layer.getBounds();
    if (bounds.isValid()) getMap().fitBounds(bounds);
  }
}
