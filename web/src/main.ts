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

  setActiveStep(id, step.title);
  const dynamic = renderStep(step);
  await step.onEnter(map, dynamic);

  // On mobile, open the sidebar automatically when a step is selected.
  document.getElementById('sidebar')?.classList.add('sidebar-open');
}

async function bootstrap(): Promise<void> {
  initMap('map');
  initSidebar(steps, (id) => void onStepSelect(id));

  setContent(`
    <div class="welcome-card">
      <h2 class="welcome-title">Rural Bus Route Planning</h2>
      <p class="welcome-body">
        A GIS pipeline for generating optimised rural bus routes in County
        Donegal, built on community detection and population-weighted graph
        analysis.
      </p>
      <p class="welcome-hint">
        Use the step navigator above to walk through the pipeline.
      </p>
    </div>
  `);

  // Print button
  document.getElementById('btn-print')?.addEventListener('click', () => window.print());

  // Mobile sidebar toggle
  document.getElementById('sidebar-toggle')?.addEventListener('click', () => {
    document.getElementById('sidebar')?.classList.toggle('sidebar-open');
  });

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
