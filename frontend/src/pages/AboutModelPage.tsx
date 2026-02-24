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
import type { ReactElement } from "react";
import {
  FITTING_DESCRIPTION,
  HOLD_TYPE_DESCRIPTIONS,
  MODEL_COMPONENTS,
  MODEL_SUMMARY,
  PARAM_DESCRIPTIONS,
} from "../constants/modelDescriptions";

const PARAM_LABELS: Record<string, string> = {
  o2_start: "O\u2082 Start",
  vo2: "VO\u2082",
  scale: "Scale",
  p50: "P50",
  n: "Hill Coefficient (n)",
  r_offset: "Residual Offset",
  r_decay: "Residual Decay",
  tau_decay: "\u03C4 Decay",
  lag: "Lag",
};

const PARAM_UNITS: Record<string, string> = {
  o2_start: "mL",
  vo2: "mL/min",
  scale: "mL/mmHg-eq.",
  p50: "mmHg-eq.",
  n: "dimensionless",
  r_offset: "% SpO\u2082",
  r_decay: "% SpO\u2082",
  tau_decay: "seconds",
  lag: "seconds",
};

const PARAM_RANGES: Record<string, string> = {
  o2_start: "400 \u2013 2,800",
  vo2: "100 \u2013 300",
  scale: "5 \u2013 50",
  p50: "15 \u2013 60",
  n: "2.0 \u2013 4.0",
  r_offset: "\u22125.0 \u2013 5.0",
  r_decay: "\u22123.0 \u2013 3.0",
  tau_decay: "10 \u2013 90",
  lag: "10 \u2013 30",
};

const COMPONENT_ICONS: Record<string, ReactElement> = {
  lungs: <LungsIcon />,
  curve: <CurveIcon />,
  correction: <CorrectionIcon />,
  lag: <LagIcon />,
};

const HOLD_TYPE_ORDER = ["FRC", "RV", "FL"] as const;

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography variant="h5" sx={{ mt: 5, mb: 2 }}>
      {children}
    </Typography>
  );
}

function EquationBlock({ equation }: { equation: string }) {
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
        fontFamily: "monospace",
        fontSize: "0.95rem",
        letterSpacing: "0.01em",
        overflowX: "auto",
      }}
    >
      {equation}
    </Box>
  );
}

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

      {/* ── Overview equations ──────────────────────────────────── */}
      <SectionTitle>Model Equations</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        The complete model is built from four layered equations. Each transforms
        the output of the previous step, from raw O\u2082 stores down to the
        final SpO\u2082 reading at the finger.
      </Typography>
      <Paper sx={{ p: 2.5 }}>
        {MODEL_SUMMARY.equations.map((eq, i) => (
          <Box key={eq.label} sx={{ mb: i < MODEL_SUMMARY.equations.length - 1 ? 2 : 0 }}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 0.5 }}>
              {i + 1}. {eq.label}
            </Typography>
            <Box
              sx={{
                fontFamily: "monospace",
                fontSize: "0.9rem",
                pl: 2,
                py: 0.5,
                borderLeft: "2px solid",
                borderColor: "primary.light",
              }}
            >
              {eq.formula}
            </Box>
          </Box>
        ))}
      </Paper>

      {/* ── Model Components ───────────────────────────────────── */}
      <SectionTitle>Model Components</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        The model decomposes SpO\u2082 prediction into four physiological
        processes. Each section below explains the component, its equation, and
        the parameters it uses.
      </Typography>

      {MODEL_COMPONENTS.map((comp) => (
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

          <EquationBlock equation={comp.equation} />

          <Typography variant="body2" sx={{ mt: 2, lineHeight: 1.7 }}>
            {comp.detail}
          </Typography>

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
      ))}

      {/* ── Parameter Reference ────────────────────────────────── */}
      <SectionTitle>Parameter Reference</SectionTitle>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        All nine fitted parameters with their units, typical ranges (across all
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
        has different initial O\u2082 stores and correspondingly different
        parameter bounds for the fit.
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
                O\u2082 start range:{" "}
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
