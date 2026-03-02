/**
 * Layer factory functions: GeoJSON data → Leaflet layers.
 *
 * Each function wraps addGeoJSONLayer with sensible defaults for a given
 * data type (nodes, edges). Steps can pass custom options to override.
 */

import L from 'leaflet';
import { addGeoJSONLayer } from './map.ts';
import type { NodeFeatureCollection, EdgeFeatureCollection } from './api.ts';

export interface NodeLayerOptions {
  color?: string;
  fillColor?: string;
  radius?: number;
  fillOpacity?: number;
  weight?: number;
}

export interface EdgeLayerOptions {
  color?: string;
  weight?: number;
  opacity?: number;
}

export function addNodeLayer(
  id: string,
  nodes: NodeFeatureCollection,
  options: NodeLayerOptions = {},
): L.GeoJSON {
  const {
    color = '#2563eb',
    fillColor,
    radius = 3,
    fillOpacity = 0.75,
    weight = 1,
  } = options;

  return addGeoJSONLayer(id, nodes, {
    pointToLayer: (_feature, latlng) =>
      L.circleMarker(latlng, {
        radius,
        color,
        fillColor: fillColor ?? color,
        weight,
        fillOpacity,
      }),
  });
}

export function addEdgeLayer(
  id: string,
  edges: EdgeFeatureCollection,
  options: EdgeLayerOptions = {},
): L.GeoJSON {
  const { color = '#64748b', weight = 1.5, opacity = 0.6 } = options;

  return addGeoJSONLayer(id, edges, {
    style: () => ({ color, weight, opacity }),
  });
}
