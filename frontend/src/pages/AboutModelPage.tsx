import {
  Air as LungsIcon,
  InfoOutlined as InfoIcon,
  Science as ScienceIcon,
  ShowChart as CurveIcon,
  Tune as CorrectionIcon,
  Timer as DelayIcon,
  Sensors as SensorIcon,
} from "@mui/icons-material";
import {
  Box,
  Chip,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from "@mui/material";
import "katex/dist/katex.min.css";
import type { Data, Layout } from "plotly.js";
import type { ReactElement } from "react";
import { useMemo } from "react";
import { BlockMath } from "react-katex";
import Plot from "react-plotly.js";
import {
  FITTING_DESCRIPTION,
  HOLD_TYPE_DESCRIPTIONS,
  MODEL_COMPONENTS,
  MODEL_SUMMARY,
  PARAM_DESCRIPTIONS,
} from "../constants/modelDescriptions";
import { chartColors, plotlyLayout } from "../theme";

const PARAM_LABELS: Record<string, string> = {
  tau_0: "\u03C4\u2080 Sensor Lag",
  p: "p (descent exponent)",
  delta: "\u03B4 (lag offset)",
  s_min: "S_min (nadir)",
  v_up: "v_up (recovery)",
  k_co2: "k_CO\u2082",
  gamma: "Steepness (\u03B3)",
  tau_washout: "\u03C4 Washout",
  paco2_0: "PACO\u2082\u2080",
  p50_base: "P50 Base (fixed)",
  pvo2_fixed: "PvO\u2082 (fixed)",
};

const PARAM_UNITS: Record<string, string> = {
  tau_0: "seconds",
  p: "dimensionless",
  delta: "seconds",
  s_min: "% SpO\u2082",
  v_up: "%/s",
  k_co2: "mmHg/s",
  gamma: "dimensionless",
  tau_washout: "seconds",
  paco2_0: "mmHg",
  p50_base: "mmHg",
  pvo2_fixed: "mmHg",
};

const PARAM_RANGES: Record<string, string> = {
  tau_0: "5 \u2013 45",
  p: "1.0 \u2013 5.0",
  delta: "-20 \u2013 20",
  s_min: "30 \u2013 100",
  v_up: "0.0 \u2013 3.0",
  k_co2: "0.02 \u2013 0.25",
  gamma: "0.8 \u2013 4.0",
  tau_washout: "10 \u2013 500",
  paco2_0: "20 \u2013 55",
  p50_base: "26.6 (fixed)",
  pvo2_fixed: "25.0 (fixed)",
};

const COMPONENT_ICONS: Record<string, ReactElement> = {
  sensor: <SensorIcon />,
  delay: <DelayIcon />,
  lungs: <LungsIcon />,
  curve: <CurveIcon />,
  correction: <CorrectionIcon />,
};

const HOLD_TYPE_ORDER = ["FRC", "RV", "FL"] as const;

const P50_BASE = 26.6;
const CHART_HEIGHT = 280;

const chartLayout: Partial<Layout> = {
  ...plotlyLayout,
  height: CHART_HEIGHT,
  margin: { l: 55, r: 20, t: 35, b: 45 },
  legend: {
    x: 1,
    xanchor: "right",
    y: 1,
    bgcolor: "rgba(255,255,255,0.85)",
    font: { size: 11 },
  },
};

function linspace(start: number, end: number, n: number): number[] {
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

// -- Chart: PAO2 washout per hold type ----------------------------------------

function PAO2WashoutChart() {
  const t = useMemo(() => linspace(0, 400, 300), []);

  const traces: Data[] = useMemo(() => {
    const holdTypes = [
      { name: "FL (\u03C4=80s)", pao2_0: 120, tau: 80, color: chartColors.spo2Fit },
      { name: "FRC (\u03C4=40s)", pao2_0: 100, tau: 40, color: chartColors.residual },
      { name: "RV (\u03C4=25s)", pao2_0: 85, tau: 25, color: chartColors.threshold },
    ];
    return holdTypes.map(({ name, pao2_0, tau, color }) => ({
      x: t,
      y: t.map((ti) => 25.0 + (pao2_0 - 25.0) * Math.exp(-ti / tau)),
      type: "scatter" as const,
      mode: "lines" as const,
      name,
      line: { color, width: 2.5 },
    }));
  }, [t]);

  return (
    <Plot
      data={traces}
      layout={{
        ...chartLayout,
        title: { text: "Alveolar PO\u2082 Washout", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "Time (s)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "PAO\u2082 (mmHg)" } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// -- Chart: Severinghaus ODC --------------------------------------------------

function SeveringhausCurveChart() {
  const pao2 = useMemo(() => linspace(0.1, 120, 300), []);

  const severinghaus = (x: number) => {
    const clamped = Math.max(x, 0.01);
    return 100.0 / (1.0 + 23400.0 / (Math.pow(clamped, 3) + 150.0 * clamped));
  };

  const severinghausGamma = (x: number, gamma: number) => {
    const clamped = Math.max(x, 0.01);
    const xAdj = P50_BASE * Math.pow(clamped / P50_BASE, gamma);
    return severinghaus(xAdj);
  };

  const traces: Data[] = useMemo(() => {
    const curves = [
      { gamma: 1.0, name: "\u03B3 = 1.0 (standard)", color: chartColors.spo2Fit, dash: "solid" as const },
      { gamma: 2.0, name: "\u03B3 = 2.0 (steeper)", color: chartColors.hr, dash: "dash" as const },
      { gamma: 3.0, name: "\u03B3 = 3.0 (very steep)", color: chartColors.threshold, dash: "dot" as const },
    ];
    const result: Data[] = curves.map(({ gamma, name, color, dash }) => ({
      x: pao2,
      y: pao2.map((p) => severinghausGamma(p, gamma)),
      type: "scatter" as const,
      mode: "lines" as const,
      name,
      line: { color, width: 2.5, dash },
    }));
    result.push({
      x: [P50_BASE, P50_BASE],
      y: [0, 50],
      type: "scatter",
      mode: "lines",
      name: `P50 \u2248 ${P50_BASE}`,
      line: { color: "rgba(0,0,0,0.3)", width: 1, dash: "dot" },
      showlegend: true,
    });
    result.push({
      x: [0, P50_BASE],
      y: [50, 50],
      type: "scatter",
      mode: "lines",
      line: { color: "rgba(0,0,0,0.3)", width: 1, dash: "dot" },
      showlegend: false,
    });
    return result;
  }, [pao2]);

  return (
    <Plot
      data={traces}
      layout={{
        ...chartLayout,
        title: { text: "Severinghaus ODC with gamma steepness", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "PaO\u2082 eff (mmHg-eq.)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "SaO\u2082 (%)" }, range: [0, 105] },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// -- Chart: Linear CO2 Bohr effect -------------------------------------------

function BohrEffectChart() {
  const t = useMemo(() => linspace(0, 400, 300), []);

  const traces: Data[] = useMemo(() => {
    const curves = [
      { k_co2: 0.0, paco2_0: 40, name: "No CO\u2082 effect", color: "rgba(0,0,0,0.3)", dash: "dot" as const },
      { k_co2: 0.04, paco2_0: 40, name: "k=0.04, mild", color: chartColors.spo2Fit, dash: "solid" as const },
      { k_co2: 0.06, paco2_0: 40, name: "k=0.06, typical", color: chartColors.hr, dash: "dash" as const },
      { k_co2: 0.10, paco2_0: 40, name: "k=0.10, strong", color: chartColors.threshold, dash: "dashdot" as const },
    ];
    return curves.map(({ k_co2, paco2_0, name, color, dash }) => ({
      x: t,
      y: t.map((ti) => P50_BASE + 0.48 * (paco2_0 + k_co2 * ti - 40)),
      type: "scatter" as const,
      mode: "lines" as const,
      name,
      line: { color, width: 2.5, dash },
    }));
  }, [t]);

  return (
    <Plot
      data={traces}
      layout={{
        ...chartLayout,
        title: { text: "Linear CO\u2082 Bohr Effect: P50 Over Time", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "Time (s)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "P50 effective (mmHg)" } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// Map component icons to charts
const COMPONENT_CHARTS: Record<string, () => ReactElement> = {
  lungs: () => <PAO2WashoutChart />,
  curve: () => <SeveringhausCurveChart />,
  correction: () => <BohrEffectChart />,
};

// -- Section helpers ----------------------------------------------------------

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography variant="h5" sx={{ mt: 5, mb: 2 }}>
      {children}
    </Typography>
  );
}

function LatexBlock({ latex }: { latex: string }) {
  return (
    <Box
      sx={{
        px: 2.5,
        py: 1.5,
        my: 1.5,
        bgcolor: "rgba(37, 99, 235, 0.04)",
        borderLeft: "3px solid",
        borderColor: "primary.main",
        borderRadius: 1,
        overflowX: "auto",
        "& .katex": { fontSize: "1.1em" },
      }}
    >
      <BlockMath math={latex} />
    </Box>
  );
}

// -- Main page ----------------------------------------------------------------

export default function AboutModelPage() {
  return (
    <Box sx={{ maxWidth: 900, pb: 6 }}>
      {/* Header */}
      <Typography variant="h4" gutterBottom>
        About the v7 Two-Stage Model
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
        {MODEL_SUMMARY.description}
      </Typography>

      {/* Overview equations */}
      <SectionTitle>Model Equations</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        The v7 model is built in two stages. Stage A calibrates the sensor response,
        Stage B fits the underlying apnea physiology with the sensor model frozen.
        The final SpO&#x2082; prediction flows through four equations.
      </Typography>
      <Paper sx={{ p: 2.5 }}>
        {MODEL_SUMMARY.equations.map((eq, i) => (
          <Box key={eq.label} sx={{ mb: i < MODEL_SUMMARY.equations.length - 1 ? 2 : 0 }}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 0.5 }}>
              {i + 1}. {eq.label}
            </Typography>
            <Box sx={{ pl: 1, "& .katex": { fontSize: "1.05em" } }}>
              <BlockMath math={eq.latex} />
            </Box>
          </Box>
        ))}
      </Paper>

      {/* Model Components */}
      <SectionTitle>Model Components</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        The model is decomposed into sensor and physiology components across two stages.
        Stage A (sensor calibration) runs first, then Stage B (apnea physiology)
        runs with the sensor model frozen.
      </Typography>

      {MODEL_COMPONENTS.map((comp) => {
        const ChartFn = COMPONENT_CHARTS[comp.icon];
        return (
          <Paper key={comp.title} sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 40,
                  height: 40,
                  borderRadius: 2,
                  bgcolor: "primary.main",
                  color: "white",
                }}
              >
                {COMPONENT_ICONS[comp.icon] ?? <ScienceIcon />}
              </Box>
              <Box>
                <Typography variant="h6" sx={{ lineHeight: 1.2 }}>
                  {comp.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {comp.summary}
                </Typography>
              </Box>
            </Box>

            <LatexBlock latex={comp.latex} />

            <Typography variant="body2" sx={{ mt: 2, mb: 2, lineHeight: 1.7 }}>
              {comp.detail}
            </Typography>

            {ChartFn && <ChartFn />}

            <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
              {comp.params.map((p) => (
                <Tooltip key={p} title={PARAM_DESCRIPTIONS[p] ?? ""} arrow>
                  <Chip
                    icon={<InfoIcon sx={{ fontSize: 14 }} />}
                    label={PARAM_LABELS[p] ?? p}
                    size="small"
                    variant="outlined"
                    sx={{ cursor: "help" }}
                  />
                </Tooltip>
              ))}
            </Box>
          </Paper>
        );
      })}

      {/* Parameter Reference */}
      <SectionTitle>Parameter Reference</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        The v7 model has global parameters shared across all holds (tau_0, p, k_co2, gamma)
        and per-hold parameters (delta, S_min, v_up, tau_washout, paco2_0), plus fixed constants.
      </Typography>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 700 }}>Parameter</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>Units</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>Range</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>Description</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(PARAM_DESCRIPTIONS).map(([key, desc]) => (
              <TableRow key={key}>
                <TableCell sx={{ fontWeight: 600, whiteSpace: "nowrap" }}>
                  {PARAM_LABELS[key] ?? key}
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace", whiteSpace: "nowrap" }}>
                  {PARAM_UNITS[key] ?? "\u2014"}
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace", whiteSpace: "nowrap" }}>
                  {PARAM_RANGES[key] ?? "\u2014"}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">{desc}</Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Hold Types */}
      <SectionTitle>Hold Types</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Breath holds are classified by the lung volume at hold onset. Each type
        has different washout time constants, with correspondingly different
        parameter bounds for tau_washout and paco2_0.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {HOLD_TYPE_ORDER.map((ht) => {
          const info = HOLD_TYPE_DESCRIPTIONS[ht];
          return (
            <Paper key={ht} sx={{ p: 2.5, flex: "1 1 250px", minWidth: 250 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                <Chip label={ht} color="primary" size="small" sx={{ fontWeight: 700 }} />
                <Typography variant="subtitle2" color="text.secondary">
                  {info.name}
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mb: 1.5, lineHeight: 1.7 }}>
                {info.description}
              </Typography>
              <Divider sx={{ mb: 1 }} />
              <Typography variant="caption" color="text.secondary">
                Bounds:{" "}
                <Typography component="span" variant="caption" sx={{ fontWeight: 600, fontFamily: "monospace" }}>
                  {info.o2Range}
                </Typography>
              </Typography>
            </Paper>
          );
        })}
      </Box>

      {/* Fitting Process */}
      <SectionTitle>{FITTING_DESCRIPTION.title}</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {FITTING_DESCRIPTION.summary}
      </Typography>

      <Paper sx={{ p: 2.5 }}>
        {FITTING_DESCRIPTION.details.map((detail, i) => (
          <Box key={i} sx={{ display: "flex", gap: 1.5, mb: i < FITTING_DESCRIPTION.details.length - 1 ? 1.5 : 0 }}>
            <Chip
              label={i + 1}
              size="small"
              color="primary"
              variant="outlined"
              sx={{ minWidth: 28, height: 24, fontSize: "0.75rem" }}
            />
            <Typography variant="body2" sx={{ lineHeight: 1.7 }}>
              {detail}
            </Typography>
          </Box>
        ))}
      </Paper>
    </Box>
  );
}
