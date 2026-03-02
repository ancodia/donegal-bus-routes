/**
 * Simple SVG bar chart.
 *
 * Returns an SVGElement that can be appended to any container.
 * Uses a responsive viewBox so it fills the container width.
 */

export interface BarData {
  label: string;
  value: number;
  color?: string;
}

export interface BarChartOptions {
  height?: number;
  formatValue?: (v: number) => string;
  barColor?: string;
}

const NS = 'http://www.w3.org/2000/svg';

function el(tag: string): SVGElement {
  return document.createElementNS(NS, tag) as SVGElement;
}

function text(
  parent: SVGElement,
  content: string,
  x: number,
  y: number,
  opts: { anchor?: string; size?: number; fill?: string } = {},
): void {
  const t = el('text') as SVGTextElement;
  t.setAttribute('x', String(x));
  t.setAttribute('y', String(y));
  t.setAttribute('text-anchor', opts.anchor ?? 'middle');
  t.setAttribute('font-size', String(opts.size ?? 10));
  t.setAttribute('fill', opts.fill ?? '#475569');
  t.setAttribute('font-family', "system-ui, -apple-system, 'Segoe UI', sans-serif");
  t.textContent = content;
  parent.appendChild(t);
}

export function createBarChart(data: BarData[], options: BarChartOptions = {}): SVGElement {
  const { height = 140, formatValue = String, barColor = '#2563eb' } = options;

  const VW = 280;
  const PAD = { top: 24, right: 8, bottom: 28, left: 8 };
  const chartW = VW - PAD.left - PAD.right;
  const chartH = height - PAD.top - PAD.bottom;
  const n = data.length;
  const barW = chartW / n;
  const GAP = Math.min(6, barW * 0.15);

  const maxVal = Math.max(...data.map((d) => d.value), 1);

  const svg = el('svg') as SVGSVGElement;
  svg.setAttribute('viewBox', `0 0 ${VW} ${height}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('aria-label', 'Bar chart');
  svg.style.display = 'block';

  const g = el('g') as SVGGElement;
  g.setAttribute('transform', `translate(${PAD.left},${PAD.top})`);
  svg.appendChild(g);

  // Baseline
  const line = el('line') as SVGLineElement;
  line.setAttribute('x1', '0');
  line.setAttribute('y1', String(chartH));
  line.setAttribute('x2', String(chartW));
  line.setAttribute('y2', String(chartH));
  line.setAttribute('stroke', '#e2e8f0');
  line.setAttribute('stroke-width', '1');
  g.appendChild(line);

  data.forEach((d, i) => {
    const barH = (d.value / maxVal) * chartH;
    const x = i * barW + GAP / 2;
    const y = chartH - barH;
    const bw = barW - GAP;

    const rect = el('rect') as SVGRectElement;
    rect.setAttribute('x', String(x));
    rect.setAttribute('y', String(y));
    rect.setAttribute('width', String(bw));
    rect.setAttribute('height', String(Math.max(barH, 1)));
    rect.setAttribute('fill', d.color ?? barColor);
    rect.setAttribute('rx', '3');
    g.appendChild(rect);

    // Value label above bar
    if (d.value > 0) {
      text(g, formatValue(d.value), x + bw / 2, y - 4, { size: 9, fill: '#374151' });
    }

    // X-axis label
    text(g, d.label, x + bw / 2, chartH + 16, { size: 9, fill: '#64748b' });
  });

  return svg;
}
