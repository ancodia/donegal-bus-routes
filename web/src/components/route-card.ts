/**
 * Route card component — clickable summary of a single generated route.
 */

export interface RouteCardData {
  community: number;
  numStops: number;
  color: string;
}

export function createRouteCard(data: RouteCardData, onClick: () => void): HTMLElement {
  const card = document.createElement('div');
  card.className = 'route-card';
  card.dataset['community'] = String(data.community);
  card.innerHTML = `
    <span class="route-card-swatch" style="background:${data.color}"></span>
    <div class="route-card-info">
      <span class="route-card-title">Route ${data.community}</span>
      <span class="route-card-meta">${data.numStops} stops</span>
    </div>
    <span class="route-card-arrow">›</span>
  `;
  card.addEventListener('click', onClick);
  return card;
}
