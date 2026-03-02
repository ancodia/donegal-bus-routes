/**
 * Sidebar DOM rendering helpers.
 */

import type { PipelineStep } from './steps/index.ts';

export function initSidebar(steps: PipelineStep[], onStepSelect: (id: number) => void): void {
  renderStepNav(steps, onStepSelect);
  setContent('<p class="placeholder">Select a step to begin.</p>');
}

export function renderStepNav(steps: PipelineStep[], onStepSelect: (id: number) => void): void {
  const nav = document.getElementById('step-nav');
  if (!nav) return;

  nav.innerHTML = '';
  for (const step of steps) {
    const btn = document.createElement('button');
    btn.className = 'step-btn';
    btn.textContent = String(step.id);
    btn.title = step.title;
    btn.setAttribute('aria-label', `Step ${step.id}: ${step.title}`);
    btn.addEventListener('click', () => onStepSelect(step.id));
    nav.appendChild(btn);
  }
}

export function setActiveStep(id: number): void {
  const nav = document.getElementById('step-nav');
  if (!nav) return;
  for (const btn of nav.querySelectorAll<HTMLButtonElement>('.step-btn')) {
    btn.classList.toggle('active', btn.textContent === String(id));
  }
}

export function setContent(html: string): void {
  const content = document.getElementById('sidebar-content');
  if (content) content.innerHTML = html;
}

export function setHealthStatus(ok: boolean): void {
  const existing = document.getElementById('health-status');
  if (existing) existing.remove();

  const badge = document.createElement('span');
  badge.id = 'health-status';
  badge.className = ok ? 'ok' : 'error';
  badge.textContent = ok ? 'API connected' : 'API unreachable';

  const content = document.getElementById('sidebar-content');
  if (content) content.prepend(badge);
}
