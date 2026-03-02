/**
 * Leaflet map initialisation and layer management.
 */

import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Donegal centre — ~54.95°N, 7.73°W
const DONEGAL_CENTRE: L.LatLngExpression = [54.95, -7.73];
const DONEGAL_ZOOM = 9;

const OSM_TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

let _map: L.Map | null = null;
const _layers = new Map<string, L.Layer>();

export function initMap(containerId: string): L.Map {
  _map = L.map(containerId, { preferCanvas: true }).setView(DONEGAL_CENTRE, DONEGAL_ZOOM);

  L.tileLayer(OSM_TILE_URL, {
    attribution: OSM_ATTRIBUTION,
    maxZoom: 19,
  }).addTo(_map);

  return _map;
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

export function fitBoundsToLayer(id: string): void {
  const layer = _layers.get(id);
  if (layer && layer instanceof L.GeoJSON) {
    const bounds = layer.getBounds();
    if (bounds.isValid()) getMap().fitBounds(bounds);
  }
}
