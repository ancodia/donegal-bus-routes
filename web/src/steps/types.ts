import type L from 'leaflet';

export interface PipelineStep {
  id: number;
  title: string;
  narrativeHtml: string;
  onEnter: (map: L.Map, sidebar: HTMLElement) => Promise<void>;
  onExit: (map: L.Map) => void;
}
