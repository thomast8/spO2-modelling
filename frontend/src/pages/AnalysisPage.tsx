import {
  Alert,
  Box,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { getDesatRate, getSensitivity, getThreshold } from "../api/analysis";
import { listAllModels, getPredictionCurve } from "../api/models";
import type { HoldType } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";

const HOLD_TYPES: HoldType[] = ["FRC", "RV", "FL"];

export default function AnalysisPage() {
  const [holdType, setHoldType] = useState<HoldType>("FL");
  const [thresholdVal, setThresholdVal] = useState(40);

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
  });

  // Find active model for selected type
  const typeVersions = models?.[holdType]?.versions ?? [];
  const activeModel = typeVersions.find((v) => v.is_active) ?? typeVersions[0];

  const { data: curve } = useQuery({
    queryKey: ["prediction", activeModel?.id],
    queryFn: () => getPredictionCurve(activeModel!.id, 800),
    enabled: !!activeModel,
  });

  const { data: thresholdResult } = useQuery({
    queryKey: ["threshold", activeModel?.id, thresholdVal],
    queryFn: () => getThreshold(activeModel!.id, thresholdVal),
    enabled: !!activeModel,
  });

  const { data: sensitivity } = useQuery({
    queryKey: ["sensitivity", activeModel?.id],
    queryFn: () => getSensitivity(activeModel!.id),
    enabled: !!activeModel,
  });

  const { data: desatRate } = useQuery({
    queryKey: ["desat-rate", activeModel?.id],
    queryFn: () => getDesatRate(activeModel!.id),
    enabled: !!activeModel,
  });

  if (!models) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Analyze fitted models: threshold prediction, VO₂ sensitivity, desaturation rates.
      </Typography>

      {/* Model selector */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12, sm: 4 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Hold Type</InputLabel>
              <Select value={holdType} label="Hold Type" onChange={(e) => setHoldType(e.target.value as HoldType)}>
                {HOLD_TYPES.map((t) => (
                  <MenuItem key={t} value={t}>{t}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 12, sm: 8 }}>
            {activeModel ? (
              <Typography variant="body2" color="text.secondary">
                Using: <strong>v{activeModel.version}</strong> (R² = {activeModel.r_squared.toFixed(4)})
              </Typography>
            ) : (
              <Alert severity="warning" sx={{ py: 0 }}>
                No model available for {holdType}. Fit one first.
              </Alert>
            )}
          </Grid>
        </Grid>
      </Paper>

      {activeModel && (
        <Grid container spacing={3}>
          {/* Prediction curve with threshold */}
          <Grid size={{ xs: 12, lg: 8 }}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Prediction Curve
              </Typography>
              {curve && (
                <SpO2Chart
                  predictedT={curve.t}
                  predictedSpo2={curve.spo2}
                  threshold={thresholdVal}
                  height={400}
                />
              )}
              <Box sx={{ px: 2, mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  SpO₂ Threshold: {thresholdVal}%
                </Typography>
                <Slider
                  value={thresholdVal}
                  onChange={(_, v) => setThresholdVal(v as number)}
                  min={20}
                  max={90}
                  step={1}
                  valueLabelDisplay="auto"
                  sx={{ maxWidth: 400 }}
                />
              </Box>
              {thresholdResult && (
                <Box sx={{ mt: 1 }}>
                  {thresholdResult.crossing_time_s ? (
                    <Typography variant="body1">
                      SpO₂ crosses <strong>{thresholdVal}%</strong> at{" "}
                      <strong>{thresholdResult.crossing_time_fmt}</strong> ({thresholdResult.crossing_time_s.toFixed(1)}s)
                    </Typography>
                  ) : (
                    <Typography variant="body1" color="text.secondary">
                      SpO₂ does not reach {thresholdVal}% within simulation range
                      {thresholdResult.spo2_at_end !== null && ` (ends at ${thresholdResult.spo2_at_end.toFixed(1)}%)`}
                    </Typography>
                  )}
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Desaturation rate */}
          <Grid size={{ xs: 12, lg: 4 }}>
            <Paper sx={{ p: 3, height: "100%" }}>
              <Typography variant="h6" gutterBottom>
                Desaturation Rate
              </Typography>
              {desatRate && (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>SpO₂</TableCell>
                        <TableCell>Rate (%/min)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {desatRate.map((r) => (
                        <TableRow key={r.time_s}>
                          <TableCell>{Math.floor(r.time_s / 60)}:{String(Math.round(r.time_s % 60)).padStart(2, "0")}</TableCell>
                          <TableCell>{r.spo2.toFixed(1)}%</TableCell>
                          <TableCell sx={{ color: r.rate_per_min < -5 ? "error.main" : "text.primary" }}>
                            {r.rate_per_min.toFixed(2)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </Grid>

          {/* VO2 Sensitivity */}
          <Grid size={{ xs: 12 }}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                VO₂ Sensitivity
              </Typography>
              {sensitivity && (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>VO₂ Change</TableCell>
                        <TableCell>VO₂ (mL/min)</TableCell>
                        <TableCell>Crossing Time</TableCell>
                        <TableCell>Margin (s)</TableCell>
                        <TableCell>SpO₂ at Ref</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sensitivity.map((s) => (
                        <TableRow
                          key={s.pct_change}
                          sx={{ bgcolor: s.pct_change === 0 ? "rgba(37,99,235,0.05)" : "transparent" }}
                        >
                          <TableCell sx={{ fontWeight: s.pct_change === 0 ? 700 : 400 }}>
                            {s.pct_change > 0 ? "+" : ""}{s.pct_change}%
                          </TableCell>
                          <TableCell>{s.vo2.toFixed(1)}</TableCell>
                          <TableCell>
                            {s.crossing_time_s
                              ? `${Math.floor(s.crossing_time_s / 60)}:${String(Math.round(s.crossing_time_s % 60)).padStart(2, "0")}`
                              : "—"}
                          </TableCell>
                          <TableCell sx={{ color: (s.margin_s ?? 0) < 0 ? "error.main" : "text.primary" }}>
                            {s.margin_s !== null ? `${s.margin_s > 0 ? "+" : ""}${s.margin_s.toFixed(1)}s` : "—"}
                          </TableCell>
                          <TableCell>{s.spo2_at_ref.toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}
