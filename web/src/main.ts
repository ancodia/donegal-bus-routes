/**
 * Application entry point.
 *
 * Initialises the Leaflet map, sidebar, and verifies API connectivity.
 * Handles step navigation: calls onExit on the previous step and onEnter on
 * the new step, passing the step-dynamic container as the sidebar target.
 */

import './style.css';

import { fetchHealth } from './api.ts';
import { initMap, getMap } from './map.ts';
import { initSidebar, setContent, setHealthStatus, setActiveStep, renderStep } from './sidebar.ts';
import { steps } from './steps/index.ts';
import type { PipelineStep } from './steps/types.ts';

let _currentStep: PipelineStep | null = null;

async function onStepSelect(id: number): Promise<void> {
  if (_currentStep?.id === id) return;

  const map = getMap();
  const step = steps.find((s) => s.id === id);
  if (!step) return;

  if (_currentStep) _currentStep.onExit(map);
  _currentStep = step;

  setActiveStep(id);
  const dynamic = renderStep(step);
  await step.onEnter(map, dynamic);
}

async function bootstrap(): Promise<void> {
  initMap('map');
  initSidebar(steps, (id) => void onStepSelect(id));

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
