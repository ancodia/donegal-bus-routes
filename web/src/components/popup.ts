/**
 * Leaflet popup HTML templates.
 */

export function createNearestStopPopup(props: {
  osmid: number;
  community: number;
  distanceKm: number;
}): string {
  return `
    <div class="nearest-popup">
      <div class="popup-title">Nearest Bus Stop</div>
      <div class="popup-row">
        <span class="popup-label">Community</span>
        <span class="popup-value">${props.community}</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">OSM ID</span>
        <span class="popup-value">${props.osmid}</span>
      </div>
      <div class="popup-row popup-highlight">
        <span class="popup-label">Distance</span>
        <span class="popup-value">~${props.distanceKm.toFixed(2)} km</span>
      </div>
    </div>
  `;
}
