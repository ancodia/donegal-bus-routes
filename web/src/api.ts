/**
 * Typed fetch wrappers for the Donegal Bus Routes API.
 * Base URL is configurable via VITE_API_URL (defaults to http://localhost:8000).
 */

// In dev, Vite proxies /api/* → localhost:8000/*.
// In production, nginx does the same (see web/nginx.conf).
// Override by setting VITE_API_URL in a .env file.
const BASE_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api';

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${path}`);
  }
  return res.json() as Promise<T>;
}

// ── GeoJSON primitives ──────────────────────────────────────────────────────

export interface PointGeometry {
  type: 'Point';
  coordinates: [number, number];
}

export interface LineStringGeometry {
  type: 'LineString';
  coordinates: [number, number][];
}

export interface GeoJSONFeature<G, P extends Record<string, unknown> | null> {
  type: 'Feature';
  geometry: G;
  properties: P;
}

export interface GeoJSONFeatureCollection<G, P extends Record<string, unknown> | null> {
  type: 'FeatureCollection';
  features: GeoJSONFeature<G, P>[];
}

// ── Graph models ────────────────────────────────────────────────────────────

export interface GraphSummary {
  num_nodes: number;
  num_edges: number;
  num_communities: number;
  is_connected: boolean;
}

export interface NodeProperties extends Record<string, unknown> {
  osmid: number;
  x: number;
  y: number;
  community: number;
  rank: number;
  top_n: number;
  route_flag: number;
}

export interface EdgeProperties extends Record<string, unknown> {
  u: number;
  v: number;
  key: number;
  length: number;
  weight: number;
  highway: string;
  name: string;
}

export type NodeFeatureCollection = GeoJSONFeatureCollection<PointGeometry, NodeProperties>;
export type EdgeFeatureCollection = GeoJSONFeatureCollection<LineStringGeometry, EdgeProperties>;

// ── Community models ────────────────────────────────────────────────────────

export interface Community {
  label: number;
  num_nodes: number;
  num_edges: number;
}

export interface CommunityCollection {
  communities: Community[];
  total: number;
}

export interface CommunityDetail {
  label: number;
  num_nodes: number;
  num_edges: number;
  top_ranked_nodes: number[];
  route_start: number | null;
  route_end: number | null;
}

// ── Route models ────────────────────────────────────────────────────────────

export interface RouteStop {
  osmid: number;
  x: number;
  y: number;
  sequence: number;
}

export interface RouteDetail {
  community: number;
  stops: RouteStop[];
  total_weight: number;
}

export interface RouteCollection {
  routes: RouteDetail[];
  total_routes: number;
}

export interface RouteFeatureProperties extends Record<string, unknown> {
  community: number;
  num_stops: number;
}

export type RouteFeatureCollection = GeoJSONFeatureCollection<LineStringGeometry, RouteFeatureProperties>;

// ── Stop models ─────────────────────────────────────────────────────────────

export type StopFeatureCollection = GeoJSONFeatureCollection<PointGeometry, Record<string, unknown>>;

// ── Analysis models ─────────────────────────────────────────────────────────

export interface AccessibilityTestResult {
  source_node: number;
  destinations_reached: number;
  destinations_total: number;
  avg_path_length: number;
}

export interface AccessibilitySummary {
  total_tests: number;
  avg_destinations_reached: number;
  avg_path_length: number;
  results: AccessibilityTestResult[];
}

export interface RouteCost {
  route_community: number;
  distance_km: number;
  estimated_fuel_litres: number;
  estimated_cost_eur: number;
}

export interface CostComparison {
  generated_routes: RouteCost[];
  actual_routes: RouteCost[];
  total_generated_cost: number;
  total_actual_cost: number;
}

export interface PopulationSummary {
  total_townlands: number;
  total_population: number;
  townlands_with_coords: number;
}

// ── Bounding box helper ─────────────────────────────────────────────────────

export interface BBox {
  minLng: number;
  minLat: number;
  maxLng: number;
  maxLat: number;
}

function bboxParams(bbox: BBox): string {
  return `?min_lng=${bbox.minLng}&min_lat=${bbox.minLat}&max_lng=${bbox.maxLng}&max_lat=${bbox.maxLat}`;
}

// ── API functions ───────────────────────────────────────────────────────────

export const fetchHealth = (): Promise<{ status: string }> =>
  apiFetch('/health');

export const fetchGraphSummary = (): Promise<GraphSummary> =>
  apiFetch('/graph/summary');

export const fetchGraphNodes = (bbox?: BBox): Promise<NodeFeatureCollection> =>
  apiFetch(`/graph/nodes${bbox ? bboxParams(bbox) : ''}`);

export const fetchGraphEdges = (bbox?: BBox): Promise<EdgeFeatureCollection> =>
  apiFetch(`/graph/edges${bbox ? bboxParams(bbox) : ''}`);

export const fetchCommunities = (): Promise<CommunityCollection> =>
  apiFetch('/communities/');

export const fetchCommunity = (label: number): Promise<CommunityDetail> =>
  apiFetch(`/communities/${label}`);

export const fetchRoutes = (): Promise<RouteCollection> =>
  apiFetch('/routes/');

export const fetchRoutesGeoJSON = (): Promise<RouteFeatureCollection> =>
  apiFetch('/routes/geojson');

export const fetchRouteByCommunity = (label: number): Promise<RouteDetail> =>
  apiFetch(`/routes/community/${label}`);

export const fetchConnectionRoute = (label: string): Promise<RouteDetail> =>
  apiFetch(`/routes/connection/${label}`);

export const fetchGeneratedStops = (): Promise<StopFeatureCollection> =>
  apiFetch('/stops/generated');

export const fetchActualStops = (): Promise<StopFeatureCollection> =>
  apiFetch('/stops/actual');

export const fetchNearestStop = (lat: number, lng: number): Promise<StopFeatureCollection> =>
  apiFetch(`/stops/nearest?lat=${lat}&lng=${lng}`);

export const fetchPopulation = (): Promise<PopulationSummary> =>
  apiFetch('/analysis/population');

export const fetchAccessibility = (): Promise<AccessibilitySummary> =>
  apiFetch('/analysis/accessibility');

export const fetchCost = (): Promise<CostComparison> =>
  apiFetch('/analysis/cost');
