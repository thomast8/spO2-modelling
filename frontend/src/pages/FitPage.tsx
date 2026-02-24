import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { previewFit, saveFit } from "../api/fit";
import { listSessions, getSession } from "../api/sessions";
import type { HoldType } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { useAppStore } from "../store/appStore";

const HOLD_TYPES: HoldType[] = ["FRC", "RV", "FL"];

const PARAM_LABELS: Record<string, string> = {
  o2_start: "O₂ Start (mL)",
  vo2: "VO₂ (mL/min)",
  scale: "Scale",
  p50: "P50 (mmHg)",
  n: "Hill Coefficient",
  r_offset: "Residual Offset",
  r_decay: "Residual Decay",
  tau_decay: "Tau Decay (s)",
  lag: "Lag (s)",
};

export default function FitPage() {
  const queryClient = useQueryClient();
  const { fitPreview, fitHoldType, fitHoldIds, setFitPreview, clearFitPreview } = useAppStore();

  const [holdType, setHoldType] = useState<HoldType>("FRC");
  const [selectedSessionId, setSelectedSessionId] = useState<number | "">("");
  const [selectedHoldIds, setSelectedHoldIds] = useState<number[]>([]);
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { data: sessions } = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
  });

  const { data: sessionDetail } = useQuery({
    queryKey: ["session", selectedSessionId],
    queryFn: () => getSession(selectedSessionId as number),
    enabled: !!selectedSessionId,
  });

  const eligibleHolds = sessionDetail?.holds.filter(
    (h) => h.include_in_fit && (h.hold_type === holdType || h.hold_type === "untagged"),
  ) ?? [];

  const fitMutation = useMutation({
    mutationFn: previewFit,
    onSuccess: (data) => {
      setFitPreview(data, holdType, selectedHoldIds);
      setError(null);
    },
    onError: (err: { response?: { data?: { detail?: string } } }) => {
      setError(err.response?.data?.detail ?? "Fit failed");
    },
  });

  const saveMutation = useMutation({
    mutationFn: saveFit,
    onSuccess: () => {
      clearFitPreview();
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const handleRunFit = () => {
    if (selectedHoldIds.length === 0) return;
    setError(null);
    fitMutation.mutate({
      hold_type: holdType,
      hold_ids: selectedHoldIds,
    });
  };

  const handleSave = () => {
    if (!fitPreview || !fitHoldType) return;
    saveMutation.mutate({
      hold_type: fitHoldType,
      params: fitPreview.params,
      hold_ids: fitHoldIds,
      r_squared: fitPreview.r_squared,
      objective_val: fitPreview.objective_val,
      converged: fitPreview.converged,
      notes: notes || undefined,
      set_active: true,
    });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Fit Model
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Select holds, run a fit, preview results, then save as a new model version.
      </Typography>

      {/* Controls */}
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
          <Grid size={{ xs: 12, sm: 4 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Session</InputLabel>
              <Select
                value={selectedSessionId}
                label="Session"
                onChange={(e) => {
                  setSelectedSessionId(e.target.value as number);
                  setSelectedHoldIds([]);
                }}
              >
                {sessions?.map((s) => (
                  <MenuItem key={s.id} value={s.id}>{s.name} ({s.session_date})</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
            <Button
              variant="contained"
              fullWidth
              onClick={handleRunFit}
              disabled={selectedHoldIds.length === 0 || fitMutation.isPending}
            >
              {fitMutation.isPending ? <CircularProgress size={22} /> : "Run Fit"}
            </Button>
          </Grid>
        </Grid>

        {/* Hold selection chips */}
        {eligibleHolds.length > 0 && (
          <Box sx={{ mt: 2, display: "flex", flexWrap: "wrap", gap: 1 }}>
            {eligibleHolds.map((h) => (
              <Chip
                key={h.id}
                label={`Hold ${h.hold_number} (${h.duration_s}s)`}
                clickable
                color={selectedHoldIds.includes(h.id) ? "primary" : "default"}
                variant={selectedHoldIds.includes(h.id) ? "filled" : "outlined"}
                onClick={() =>
                  setSelectedHoldIds((prev) =>
                    prev.includes(h.id) ? prev.filter((id) => id !== h.id) : [...prev, h.id],
                  )
                }
              />
            ))}
            <Button
              size="small"
              onClick={() => setSelectedHoldIds(eligibleHolds.map((h) => h.id))}
            >
              Select All
            </Button>
          </Box>
        )}
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Fit Preview */}
      {fitPreview && (
        <Box>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Fit Preview — {fitHoldType}</Typography>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Chip
                  label={`R² = ${fitPreview.r_squared.toFixed(4)}`}
                  color={fitPreview.r_squared > 0.95 ? "success" : fitPreview.r_squared > 0.9 ? "warning" : "error"}
                />
                <Chip label={fitPreview.converged ? "Converged" : "Not converged"} color={fitPreview.converged ? "success" : "warning"} variant="outlined" />
              </Box>
            </Box>

            {/* Prediction charts per hold */}
            {fitPreview.predictions.map((pred) => (
              <Box key={pred.hold_id} sx={{ mb: 2 }}>
                <SpO2Chart
                  observedT={pred.elapsed_s}
                  observedSpo2={pred.observed}
                  predictedT={pred.elapsed_s}
                  predictedSpo2={pred.predicted}
                  title={`Hold ${pred.hold_id} — R² = ${pred.r_squared.toFixed(4)}`}
                  height={300}
                />
              </Box>
            ))}

            {/* Parameters table */}
            <TableContainer>
              <Table size="small">
                <TableBody>
                  {Object.entries(fitPreview.params).map(([key, val]) => (
                    <TableRow key={key}>
                      <TableCell sx={{ fontWeight: 600, border: "none" }}>
                        {PARAM_LABELS[key] ?? key}
                      </TableCell>
                      <TableCell sx={{ border: "none", fontFamily: "monospace" }}>
                        {typeof val === "number" ? val.toFixed(4) : val}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Save controls */}
          <Paper sx={{ p: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid size={{ xs: 12, sm: 8 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Notes (optional)"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="e.g., Initial FL model from hold 6"
                />
              </Grid>
              <Grid size={{ xs: 6, sm: 2 }}>
                <Button
                  variant="contained"
                  color="success"
                  fullWidth
                  onClick={handleSave}
                  disabled={saveMutation.isPending}
                >
                  {saveMutation.isPending ? <CircularProgress size={22} /> : "Save"}
                </Button>
              </Grid>
              <Grid size={{ xs: 6, sm: 2 }}>
                <Button
                  variant="outlined"
                  color="error"
                  fullWidth
                  onClick={clearFitPreview}
                >
                  Discard
                </Button>
              </Grid>
            </Grid>
            {saveMutation.isSuccess && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Model saved successfully!
              </Alert>
            )}
          </Paper>
        </Box>
      )}
    </Box>
  );
}
