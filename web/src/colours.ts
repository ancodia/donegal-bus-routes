/**
 * Categorical colour palette for 14 communities.
 * Based on Tableau 20 — perceptually distinct at typical map zoom levels.
 */

const PALETTE: readonly string[] = [
  '#4e79a7', // 0  — steel blue
  '#f28e2b', // 1  — orange
  '#e15759', // 2  — red
  '#76b7b2', // 3  — teal
  '#59a14f', // 4  — green
  '#edc948', // 5  — yellow
  '#b07aa1', // 6  — purple
  '#ff9da7', // 7  — pink
  '#9c755f', // 8  — brown
  '#bab0ac', // 9  — grey
  '#d37295', // 10 — deep pink
  '#a0cbe8', // 11 — light blue
  '#86bcb6', // 12 — seafoam
  '#8cd17d', // 13 — light green
];

const FALLBACK = '#94a3b8';

export function communityColour(label: number): string {
  return PALETTE[label % PALETTE.length] ?? FALLBACK;
}

export { PALETTE as COMMUNITY_PALETTE };
