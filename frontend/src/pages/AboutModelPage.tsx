import {
  Air as LungsIcon,
  InfoOutlined as InfoIcon,
  Science as ScienceIcon,
  ShowChart as CurveIcon,
  Timer as LagIcon,
  Tune as CorrectionIcon,
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
  pao2_0: "PAO\u2082 Initial",
  pvo2: "PvO\u2082",
  tau_washout: "\u03C4 Washout",
  p50_base: "P50 Base",
  n: "Hill Coefficient (n)",
  bohr_coeff: "Bohr Coeff",
  lag: "Lag",
  r_offset: "Offset",
};

const PARAM_UNITS: Record<string, string> = {
  pao2_0: "mmHg",
  pvo2: "mmHg",
  tau_washout: "seconds",
  p50_base: "mmHg",
  n: "dimensionless",
  bohr_coeff: "mmHg/s",
  lag: "seconds",
  r_offset: "% SpO\u2082",
};

const PARAM_RANGES: Record<string, string> = {
  pao2_0: "70 \u2013 145",
  pvo2: "30 \u2013 50",
  tau_washout: "10 \u2013 250",
  p50_base: "22 \u2013 32",
  n: "2.0 \u2013 4.0",
  bohr_coeff: "0.0 \u2013 0.10",
  lag: "5 \u2013 30",
  r_offset: "\u22123.0 \u2013 3.0",
};

const COMPONENT_ICONS: Record<string, ReactElement> = {
  lungs: <LungsIcon />,
  curve: <CurveIcon />,
  correction: <CorrectionIcon />,
  lag: <LagIcon />,
};

const HOLD_TYPE_ORDER = ["FRC", "RV", "FL"] as const;

// Default parameters for demonstration charts (typical FL hold)
const DEMO_PARAMS = {
  pao2_0: 120,
  pvo2: 40,
  tau_washout: 80,
  p50_base: 26.6,
  n: 2.7,
  bohr_coeff: 0.02,
  lag: 19,
  r_offset: 0,
};

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

// ── Helper: generate time array ──────────────────────────────
function linspace(start: number, end: number, n: number): number[] {
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

// ── Chart 1: PAO2 washout over time ─────────────────────────
function PAO2WashoutChart() {
  const t = useMemo(() => linspace(0, 400, 300), []);

  const traces: Data[] = useMemo(() => {
    const holdTypes = [
      { name: "FL (\u03C4=80s, PAO\u2082\u2080=120)", pao2_0: 120, tau: 80, color: chartColors.spo2Fit },
      { name: "FRC (\u03C4=40s, PAO\u2082\u2080=100)", pao2_0: 100, tau: 40, color: chartColors.residual },
      { name: "RV (\u03C4=25s, PAO\u2082\u2080=85)", pao2_0: 85, tau: 25, color: chartColors.threshold },
    ];
    return holdTypes.map(({ name, pao2_0, tau, color }) => ({
      x: t,
      y: t.map((ti) => {
        const tEff = Math.max(ti - DEMO_PARAMS.lag, 0);
        return DEMO_PARAMS.pvo2 + (pao2_0 - DEMO_PARAMS.pvo2) * Math.exp(-tEff / tau);
      }),
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

// ── Chart 2: Hill dissociation curve ─────────────────────────
function HillCurveChart() {
  const pao2 = useMemo(() => linspace(0.1, 120, 300), []);

  const traces: Data[] = useMemo(() => {
    const hillCoeffs = [
      { n: 2.0, name: "n = 2.0", color: chartColors.residual, dash: "dot" as const },
      { n: 2.7, name: "n = 2.7 (physiological)", color: chartColors.spo2Fit, dash: "solid" as const },
      { n: 4.0, name: "n = 4.0", color: chartColors.hr, dash: "dash" as const },
    ];
    const result: Data[] = hillCoeffs.map(({ n, name, color, dash }) => ({
      x: pao2,
      y: pao2.map((p) => 100 * Math.pow(p, n) / (Math.pow(p, n) + Math.pow(DEMO_PARAMS.p50_base, n))),
      type: "scatter" as const,
      mode: "lines" as const,
      name,
      line: { color, width: 2.5, dash },
    }));
    // P50 marker line
    result.push({
      x: [DEMO_PARAMS.p50_base, DEMO_PARAMS.p50_base],
      y: [0, 50],
      type: "scatter",
      mode: "lines",
      name: `P50 = ${DEMO_PARAMS.p50_base}`,
      line: { color: "rgba(0,0,0,0.3)", width: 1, dash: "dot" },
      showlegend: true,
    });
    result.push({
      x: [0, DEMO_PARAMS.p50_base],
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
        title: { text: "Oxygen\u2013Haemoglobin Dissociation Curve", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "PaO\u2082 eff (mmHg-eq.)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "SpO\u2082 (%)" }, range: [0, 105] },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// ── Chart 3: Bohr effect — P50 shift over time ──────────────
function BohrEffectChart() {
  const t = useMemo(() => linspace(0, 400, 300), []);

  const traces: Data[] = useMemo(() => {
    const bohrCoeffs = [
      { beta: 0.0, name: "\u03B2 = 0 (no Bohr)", color: "rgba(0,0,0,0.3)", dash: "dot" as const },
      { beta: 0.02, name: "\u03B2 = 0.02 (typical)", color: chartColors.spo2Fit, dash: "solid" as const },
      { beta: 0.05, name: "\u03B2 = 0.05 (strong)", color: chartColors.threshold, dash: "dash" as const },
    ];
    return bohrCoeffs.map(({ beta, name, color, dash }) => ({
      x: t,
      y: t.map((ti) => {
        const tEff = Math.max(ti - DEMO_PARAMS.lag, 0);
        return DEMO_PARAMS.p50_base + beta * tEff;
      }),
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
        title: { text: "Bohr Effect: P50 Shift Over Time", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "Time (s)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "P50 effective (mmHg)" } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// ── Chart 4: Lag effect comparison ───────────────────────────
function LagChart() {
  const t = useMemo(() => linspace(0, 400, 300), []);

  const computeSpo2 = useMemo(() => (ti: number, lag: number) => {
    const tEff = Math.max(ti - lag, 0);
    const pao2 = DEMO_PARAMS.pvo2 + (DEMO_PARAMS.pao2_0 - DEMO_PARAMS.pvo2) * Math.exp(-tEff / DEMO_PARAMS.tau_washout);
    const p50Eff = DEMO_PARAMS.p50_base + DEMO_PARAMS.bohr_coeff * tEff;
    const base = 100 * Math.pow(pao2, DEMO_PARAMS.n) /
      (Math.pow(pao2, DEMO_PARAMS.n) + Math.pow(p50Eff, DEMO_PARAMS.n));
    return Math.min(Math.max(base + DEMO_PARAMS.r_offset, 0), 100);
  }, []);

  const traces: Data[] = useMemo(() => [
    {
      x: t,
      y: t.map((ti) => computeSpo2(ti, 0)),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "lag = 0 s (arterial)",
      line: { color: "rgba(0,0,0,0.3)", width: 1.5, dash: "dot" as const },
    },
    {
      x: t,
      y: t.map((ti) => computeSpo2(ti, 12)),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "lag = 12 s",
      line: { color: chartColors.spo2Fit, width: 2.5 },
    },
    {
      x: t,
      y: t.map((ti) => computeSpo2(ti, 25)),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "lag = 25 s",
      line: { color: chartColors.threshold, width: 2, dash: "dash" as const },
    },
  ], [t, computeSpo2]);

  return (
    <Plot
      data={traces}
      layout={{
        ...chartLayout,
        title: { text: "Effect of Finger\u2013Arterial Lag", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "Time (s)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "SpO\u2082 (%)" }, range: [0, 105] },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: CHART_HEIGHT }}
    />
  );
}

// ── Chart for full model ─────────────────────────────────────
function FullModelChart() {
  const t = useMemo(() => linspace(0, 400, 400), []);

  const traces: Data[] = useMemo(() => {
    const p = DEMO_PARAMS;
    const pao2Arr = t.map((ti) => {
      const tEff = Math.max(ti - p.lag, 0);
      return p.pvo2 + (p.pao2_0 - p.pvo2) * Math.exp(-tEff / p.tau_washout);
    });
    const spo2 = t.map((ti, i) => {
      const tEff = Math.max(ti - p.lag, 0);
      const pao2 = pao2Arr[i];
      const p50Eff = p.p50_base + p.bohr_coeff * tEff;
      const base = 100 * Math.pow(pao2, p.n) / (Math.pow(pao2, p.n) + Math.pow(p50Eff, p.n));
      return Math.min(Math.max(base + p.r_offset, 0), 100);
    });

    return [
      {
        x: t, y: pao2Arr,
        type: "scatter" as const, mode: "lines" as const,
        name: "PAO\u2082 (mmHg)",
        line: { color: chartColors.residual, width: 1.5, dash: "dash" as const },
        yaxis: "y2",
      },
      {
        x: t, y: spo2,
        type: "scatter" as const, mode: "lines" as const,
        name: "SpO\u2082(t)",
        line: { color: chartColors.spo2Fit, width: 3 },
      },
    ];
  }, [t]);

  return (
    <Plot
      data={traces}
      layout={{
        ...chartLayout,
        height: 320,
        title: { text: "Complete Model Output (FL hold, typical parameters)", font: { size: 13 } },
        xaxis: { ...chartLayout.xaxis, title: { text: "Time (s)" } },
        yaxis: { ...chartLayout.yaxis, title: { text: "SpO\u2082 (%)" }, range: [0, 105] },
        yaxis2: {
          title: { text: "PAO\u2082 (mmHg)" },
          overlaying: "y",
          side: "right",
          showgrid: false,
          range: [30, 130],
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: 320 }}
    />
  );
}

// Map component icons to their charts
const COMPONENT_CHARTS: Record<string, () => ReactElement> = {
  lungs: () => <PAO2WashoutChart />,
  curve: () => <HillCurveChart />,
  correction: () => <BohrEffectChart />,
  lag: () => <LagChart />,
};

// ── Section helpers ──────────────────────────────────────────

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

// ── Main page ────────────────────────────────────────────────

export default function AboutModelPage() {
  return (
    <Box sx={{ maxWidth: 900, pb: 6 }}>
      {/* ── Header ──────────────────────────────────────────────── */}
      <Typography variant="h4" gutterBottom>
        About the Model
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
        {MODEL_SUMMARY.description}
      </Typography>

      {/* ── Full model overview chart ──────────────────────────── */}
      <Paper sx={{ p: 2, mt: 3 }}>
        <FullModelChart />
      </Paper>

      {/* ── Overview equations ──────────────────────────────────── */}
      <SectionTitle>Model Equations</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        The complete model is built from four layered equations. Each transforms
        the output of the previous step, from alveolar PO&#x2082; washout down to the
        final SpO&#x2082; reading at the finger.
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

      {/* ── Model Components ───────────────────────────────────── */}
      <SectionTitle>Model Components</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        The model decomposes SpO&#x2082; prediction into four physiological
        processes. Each section below explains the component, its equation, and
        includes an interactive chart showing how the component behaves.
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

            {/* Interactive chart for this component */}
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

      {/* ── Parameter Reference ────────────────────────────────── */}
      <SectionTitle>Parameter Reference</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        All eight fitted parameters with their units, typical ranges (across all
        hold types), and physical interpretation.
      </Typography>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 700 }}>Parameter</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>Units</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>Typical Range</TableCell>
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

      {/* ── Hold Types ─────────────────────────────────────────── */}
      <SectionTitle>Hold Types</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Breath holds are classified by the lung volume at hold onset. Each type
        has different initial PAO&#x2082; and washout time constants, with correspondingly
        different parameter bounds for the fit.
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
                Initial PAO&#x2082; range:{" "}
                <Typography component="span" variant="caption" sx={{ fontWeight: 600, fontFamily: "monospace" }}>
                  {info.o2Range}
                </Typography>
              </Typography>
            </Paper>
          );
        })}
      </Box>

      {/* ── Fitting Process ────────────────────────────────────── */}
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
