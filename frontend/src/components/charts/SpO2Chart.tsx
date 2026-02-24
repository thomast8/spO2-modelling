import Plot from "react-plotly.js";
import { chartColors, plotlyLayout, spo2Colorscale } from "../../theme";
import type { Data, Layout } from "plotly.js";

interface SpO2ChartProps {
  /** Time points for observed data */
  observedT?: number[];
  /** Observed SpO2 values */
  observedSpo2?: number[];
  /** Time points for predicted curve */
  predictedT?: number[];
  /** Predicted SpO2 values */
  predictedSpo2?: number[];
  /** Optional threshold line */
  threshold?: number;
  /** Chart title */
  title?: string;
  /** Chart height */
  height?: number;
  /** Additional Plotly traces */
  extraTraces?: Data[];
  /** Additional layout overrides */
  layoutOverrides?: Partial<Layout>;
}

export default function SpO2Chart({
  observedT,
  observedSpo2,
  predictedT,
  predictedSpo2,
  threshold,
  title,
  height = 350,
  extraTraces = [],
  layoutOverrides = {},
}: SpO2ChartProps) {
  const traces: Data[] = [];

  if (observedT && observedSpo2) {
    traces.push({
      x: observedT,
      y: observedSpo2,
      type: "scatter",
      mode: "lines+markers",
      name: "Observed",
      marker: {
        color: observedSpo2,
        colorscale: spo2Colorscale,
        cmin: 70,
        cmax: 100,
        size: 5,
        showscale: true,
        colorbar: {
          title: { text: "SpO₂ %", font: { size: 11 } },
          thickness: 12,
          len: 0.6,
          tickfont: { size: 10 },
        },
      },
      line: {
        color: "rgba(0,0,0,0.1)",
        width: 1,
      },
    });
  }

  if (predictedT && predictedSpo2) {
    traces.push({
      x: predictedT,
      y: predictedSpo2,
      type: "scatter",
      mode: "lines",
      name: "Model Fit",
      line: {
        color: chartColors.spo2Fit,
        width: 2.5,
      },
    });
  }

  if (threshold !== undefined) {
    const xRange = [
      0,
      Math.max(
        ...(observedT ?? [0]),
        ...(predictedT ?? [0]),
        300,
      ),
    ];
    traces.push({
      x: xRange,
      y: [threshold, threshold],
      type: "scatter",
      mode: "lines",
      name: `Threshold ${threshold}%`,
      line: {
        color: chartColors.threshold,
        width: 1.5,
        dash: "dash",
      },
    });
  }

  traces.push(...extraTraces);

  const layout: Partial<Layout> = {
    ...plotlyLayout,
    title: title ? { text: title, font: { size: 14 } } : undefined,
    xaxis: {
      ...plotlyLayout.xaxis,
      title: { text: "Time (s)" },
    },
    yaxis: {
      ...plotlyLayout.yaxis,
      title: { text: "SpO₂ (%)" },
      range: [0, 105],
    },
    legend: {
      x: 1,
      xanchor: "right",
      y: 1,
      bgcolor: "rgba(255,255,255,0.85)",
      font: { size: 11 },
    },
    height,
    ...layoutOverrides,
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        responsive: true,
      }}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
