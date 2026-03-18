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
import SpO2Chart from "../components/charts/SpO2Chart";

export default function AnalysisPage() {
  const [thresholdVal, setThresholdVal] = useState(40);

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
  });

  // Find active v7 model
  const activeModel = models?.versions.find((v) => v.is_active) ?? models?.versions[0];

  // Hold selector (v7 predictions are per-hold)
  const holdIds = activeModel?.hold_ids ?? [];
  const [selectedHoldId, setSelectedHoldId] = useState<number | null>(null);
  const holdId = selectedHoldId ?? holdIds[0] ?? null;

  const { data: curve } = useQuery({
    queryKey: ["prediction", activeModel?.id, holdId],
    queryFn: () => getPredictionCurve(activeModel!.id, holdId!, 800),
    enabled: !!activeModel && holdId !== null,
  });

  const { data: thresholdResult } = useQuery({
    queryKey: ["threshold", activeModel?.id, holdId, thresholdVal],
    queryFn: () => getThreshold(activeModel!.id, holdId!, thresholdVal),
    enabled: !!activeModel && holdId !== null,
  });

  const { data: sensitivity } = useQuery({
    queryKey: ["sensitivity", activeModel?.id, holdId],
    queryFn: () => getSensitivity(activeModel!.id, holdId!),
    enabled: !!activeModel && holdId !== null,
  });

  const { data: desatRate } = useQuery({
    queryKey: ["desat-rate", activeModel?.id, holdId],
    queryFn: () => getDesatRate(activeModel!.id, holdId!),
    enabled: !!activeModel && holdId !== null,
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
        Analyze the active v7 model: threshold prediction, sensitivity, desaturation rates.
      </Typography>

      {/* Model + hold selector */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12, sm: 4 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Hold</InputLabel>
              <Select
                value={holdId ?? ""}
                label="Hold"
                onChange={(e) => setSelectedHoldId(Number(e.target.value))}
              >
                {holdIds.map((id) => (
                  <MenuItem key={id} value={id}>Hold #{id}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 12, sm: 8 }}>
            {activeModel ? (
              <Typography variant="body2" color="text.secondary">
                Using: <strong>v{activeModel.version}</strong> (R\u00B2 = {activeModel.r_squared.toFixed(4)})
              </Typography>
            ) : (
              <Alert severity="warning" sx={{ py: 0 }}>
                No v7 model available. Fit one first.
              </Alert>
            )}
          </Grid>
        </Grid>
      </Paper>

      {activeModel && holdId !== null && (
        <Grid container spacing={3}>
          {/* Prediction curve with threshold */}
          <Grid size={{ xs: 12, lg: 8 }}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Prediction Curve - Hold #{holdId}
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
                  SpO2 Threshold: {thresholdVal}%
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
                      SpO2 crosses <strong>{thresholdVal}%</strong> at{" "}
                      <strong>{thresholdResult.crossing_time_fmt}</strong> ({thresholdResult.crossing_time_s.toFixed(1)}s)
                    </Typography>
                  ) : (
                    <Typography variant="body1" color="text.secondary">
                      SpO2 does not reach {thresholdVal}% within simulation range
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
                        <TableCell>SpO2</TableCell>
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

          {/* Sensitivity */}
          <Grid size={{ xs: 12 }}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                tau_washout Sensitivity
              </Typography>
              {sensitivity && (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>% Change</TableCell>
                        <TableCell>Param Value</TableCell>
                        <TableCell>Crossing Time</TableCell>
                        <TableCell>Margin (s)</TableCell>
                        <TableCell>SpO2 at Ref</TableCell>
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
                          <TableCell>{s.param_value.toFixed(2)}</TableCell>
                          <TableCell>
                            {s.crossing_time_s
                              ? `${Math.floor(s.crossing_time_s / 60)}:${String(Math.round(s.crossing_time_s % 60)).padStart(2, "0")}`
                              : "\u2014"}
                          </TableCell>
                          <TableCell sx={{ color: (s.margin_s ?? 0) < 0 ? "error.main" : "text.primary" }}>
                            {s.margin_s !== null ? `${s.margin_s > 0 ? "+" : ""}${s.margin_s.toFixed(1)}s` : "\u2014"}
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
