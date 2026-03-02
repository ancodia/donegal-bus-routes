/**
 * Application entry point.
 *
 * Initialises the Leaflet map, sidebar, and verifies API connectivity.
 * Step navigation content is added in subsequent phases.
 */

import './style.css';

import { fetchHealth } from './api.ts';
import { initMap } from './map.ts';
import { initSidebar, setContent, setHealthStatus } from './sidebar.ts';
import { steps } from './steps/index.ts';

function onStepSelect(_id: number): void {
  // Step navigation handled in Phase 2+
}

async function bootstrap(): Promise<void> {
  initMap('map');
  initSidebar(steps, onStepSelect);

  setContent(`
    <h2 style="font-size:1rem;font-weight:600;margin-bottom:.5rem;">
      Rural Bus Route Planning
    </h2>
    <p style="color:#475569;line-height:1.6;">
      A GIS pipeline for generating optimised rural bus routes in County Donegal,
      built on community detection and population-weighted graph analysis.
    </p>
    <p style="color:#94a3b8;font-size:.8rem;margin-top:.75rem;">
      Use the step navigator above to walk through the pipeline.
    </p>
  `);

  try {
    const health = await fetchHealth();
    console.info('[API] health:', health);
    setHealthStatus(health.status === 'ok');
  } catch (err) {
    console.error('[API] health check failed:', err);
    setHealthStatus(false);
  }
}

bootstrap().catch(console.error);
