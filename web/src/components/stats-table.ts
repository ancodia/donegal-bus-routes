/**
 * Key-value stats table component.
 */

export interface StatRow {
  label: string;
  value: string;
  highlight?: boolean;
}

export function createStatsTable(rows: StatRow[]): HTMLElement {
  const table = document.createElement('div');
  table.className = 'stats-table';

  for (const row of rows) {
    const item = document.createElement('div');
    item.className = `stats-row${row.highlight ? ' highlight' : ''}`;
    item.innerHTML = `
      <span class="stats-label">${row.label}</span>
      <span class="stats-value">${row.value}</span>
    `;
    table.appendChild(item);
  }

  return table;
}
