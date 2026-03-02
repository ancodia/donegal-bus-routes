/**
 * Loading spinner component.
 *
 * Usage:
 *   const hide = showLoading(container, 'Fetching data…');
 *   // ... await work ...
 *   hide();
 */

export function showLoading(container: HTMLElement, message = 'Loading…'): () => void {
  const el = document.createElement('div');
  el.className = 'loading-state';
  el.innerHTML = `<span class="loading-spinner" aria-hidden="true"></span>${message}`;
  container.appendChild(el);
  return () => el.remove();
}
