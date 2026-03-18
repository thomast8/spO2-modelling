import { CloudUpload as UploadIcon, Delete as DeleteIcon, InfoOutlined as InfoIcon, Science as FitIcon } from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useSearchParams } from "react-router-dom";
import { pollFitTask, saveFit, submitFitPreview } from "../api/fit";
import { getHold, updateHold } from "../api/holds";
import { getHoldPredictions, listAllModels } from "../api/models";
import { deleteSession, listSessions, getSession, uploadSession } from "../api/sessions";
import type {
  HoldSummary,
  SessionResponse,
  V7FitPrediction,
  V7FitPreviewResponse,
} from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { PARAM_DESCRIPTIONS } from "../constants/modelDescriptions";

const HOLD_TYPE_OPTIONS = ["untagged", "FRC", "RV", "FL"] as const;
const HOLD_TYPE_LABELS: Record<string, string> = { untagged: "Skip" };

// -- Hold card ----------------------------------------------------------------

function HoldCard({
  hold,
  prediction,
  onTagChange,
}: {
  hold: HoldSummary;
  prediction?: V7FitPrediction;
  onTagChange?: (holdId: number, newType: string) => void;
}) {
  const queryClient = useQueryClient();

  const { data: holdDetail } = useQuery({
    queryKey: ["hold", hold.id],
    queryFn: () => getHold(hold.id),
  });

  const updateMutation = useMutation({
    mutationFn: (update: { hold_type?: string }) =>
      updateHold(hold.id, update),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["hold", hold.id] });
      queryClient.invalidateQueries({ queryKey: ["session"] });
      if (variables.hold_type && onTagChange) onTagChange(hold.id, variables.hold_type);
    },
  });

  const currentType = holdDetail?.hold_type ?? hold.hold_type;
  const isTagged = currentType !== "untagged";

  return (
    <Paper
      sx={{
        p: 2,
        mb: 2,
        border: isTagged ? "2px solid" : "2px solid transparent",
        borderColor: isTagged ? "primary.main" : "transparent",
        transition: "border-color 0.2s",
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="h6" sx={{ fontSize: "1rem" }}>
            Hold {hold.hold_number}
          </Typography>
          <Chip label={`${hold.duration_s}s`} size="small" color="primary" variant="outlined" />
          {hold.min_spo2 !== null && (
            <Chip
              label={`Min SpO\u2082: ${hold.min_spo2}%`}
              size="small"
              color={hold.min_spo2 < 60 ? "error" : hold.min_spo2 < 80 ? "warning" : "default"}
              variant="outlined"
            />
          )}
          {prediction && (
            <Chip
              label={`R\u00B2 = ${prediction.r_squared.toFixed(4)}`}
              size="small"
              color={prediction.r_squared > 0.95 ? "success" : prediction.r_squared > 0.9 ? "warning" : "error"}
            />
          )}
        </Box>
        <Box sx={{ display: "flex", gap: 0.5 }}>
          {HOLD_TYPE_OPTIONS.map((t) => (
            <Chip
              key={t}
              label={HOLD_TYPE_LABELS[t] ?? t}
              size="small"
              variant={currentType === t ? "filled" : "outlined"}
              color={currentType === t ? (t === "untagged" ? "default" : "primary") : "default"}
              onClick={() => updateMutation.mutate({ hold_type: t })}
              sx={{ cursor: "pointer", fontWeight: currentType === t ? 700 : 400 }}
            />
          ))}
        </Box>
      </Box>
      {holdDetail?.data_points && (
        <SpO2Chart
          observedT={holdDetail.data_points.map((dp) => dp.elapsed_s)}
          observedSpo2={holdDetail.data_points.map((dp) => dp.spo2)}
          predictedT={prediction?.elapsed_s}
          predictedSpo2={prediction?.predicted}
          height={200}
        />
      )}
    </Paper>
  );
}

// -- v7 Fit result card -------------------------------------------------------

function V7FitResultCard({
  result,
  notes,
  onNotesChange,
  onSave,
  onDiscard,
  saving,
  saved,
}: {
  result: V7FitPreviewResponse;
  notes: string;
  onNotesChange: (v: string) => void;
  onSave: () => void;
  onDiscard: () => void;
  saving: boolean;
  saved: boolean;
}) {
  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Typography variant="h6">v7 Fit Results</Typography>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Chip
            label={`R\u00B2 = ${result.overall_r2.toFixed(4)}`}
            color={result.overall_r2 > 0.95 ? "success" : result.overall_r2 > 0.9 ? "warning" : "error"}
          />
          <Chip
            label={result.stage_a.converged && result.stage_b.converged ? "Both stages converged" : "Convergence warning"}
            color={result.stage_a.converged && result.stage_b.converged ? "success" : "warning"}
            variant="outlined"
          />
          <Chip label={`${result.hold_ids.length} holds`} size="small" variant="outlined" />
        </Box>
      </Box>

      {/* Stage A summary */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: "text.secondary" }}>
        Stage A: Sensor Calibration
      </Typography>
      <TableContainer sx={{ mb: 2 }}>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell sx={{ fontWeight: 600, border: "none", py: 0.5 }}>
                <Tooltip title="Base sensor lag in seconds" arrow>
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
                    tau_0 (s) <InfoIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                  </Box>
                </Tooltip>
              </TableCell>
              <TableCell sx={{ border: "none", fontFamily: "monospace", py: 0.5 }}>
                {result.stage_a.tau_0.toFixed(2)}
              </TableCell>
              <TableCell sx={{ fontWeight: 600, border: "none", py: 0.5 }}>
                <Tooltip title="Power-law descent exponent" arrow>
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
                    p <InfoIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                  </Box>
                </Tooltip>
              </TableCell>
              <TableCell sx={{ border: "none", fontFamily: "monospace", py: 0.5 }}>
                {result.stage_a.p.toFixed(2)}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>

      <TableContainer sx={{ mb: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Hold</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>delta (s)</TableCell>
              <TableCell>S_min</TableCell>
              <TableCell>v_up</TableCell>
              <TableCell>Eff. Lag (s)</TableCell>
              <TableCell>R\u00B2</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {result.stage_a.per_hold.map((ph) => (
              <TableRow key={ph.hold_id}>
                <TableCell>#{ph.hold_id}</TableCell>
                <TableCell>{ph.hold_type}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.delta.toFixed(2)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.s_min.toFixed(1)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.v_up.toFixed(3)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.eff_lag.toFixed(1)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.r_squared.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Stage B summary */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: "text.secondary" }}>
        Stage B: Apnea Physiology
      </Typography>
      <TableContainer sx={{ mb: 2 }}>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell sx={{ fontWeight: 600, border: "none", py: 0.5 }}>
                <Tooltip title={PARAM_DESCRIPTIONS["k_co2"] ?? "CO2 accumulation rate"} arrow>
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
                    k_co2 (mmHg/s) <InfoIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                  </Box>
                </Tooltip>
              </TableCell>
              <TableCell sx={{ border: "none", fontFamily: "monospace", py: 0.5 }}>
                {result.stage_b.k_co2.toFixed(4)}
              </TableCell>
              <TableCell sx={{ fontWeight: 600, border: "none", py: 0.5 }}>
                <Tooltip title={PARAM_DESCRIPTIONS["gamma"] ?? "ODC steepness"} arrow>
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
                    gamma <InfoIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                  </Box>
                </Tooltip>
              </TableCell>
              <TableCell sx={{ border: "none", fontFamily: "monospace", py: 0.5 }}>
                {result.stage_b.gamma.toFixed(4)}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>

      <TableContainer sx={{ mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Hold</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>tau_washout (s)</TableCell>
              <TableCell>paco2_0 (mmHg)</TableCell>
              <TableCell>R\u00B2</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {result.stage_b.per_hold.map((ph) => (
              <TableRow key={ph.hold_id}>
                <TableCell>#{ph.hold_id}</TableCell>
                <TableCell>{ph.hold_type}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.tau_washout.toFixed(2)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.paco2_0.toFixed(2)}</TableCell>
                <TableCell sx={{ fontFamily: "monospace" }}>{ph.r_squared.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Divider sx={{ mb: 2 }} />

      <Grid container spacing={2} alignItems="center">
        <Grid size={{ xs: 12, sm: 7 }}>
          <TextField
            fullWidth
            size="small"
            label="Notes (optional)"
            value={notes}
            onChange={(e) => onNotesChange(e.target.value)}
            placeholder="e.g., v7 two-stage model from session data"
          />
        </Grid>
        <Grid size={{ xs: 6, sm: 2.5 }}>
          <Button
            variant="contained"
            color="success"
            fullWidth
            onClick={onSave}
            disabled={saving || saved}
          >
            {saving ? <CircularProgress size={22} /> : saved ? "Saved" : "Save Model"}
          </Button>
        </Grid>
        <Grid size={{ xs: 6, sm: 2.5 }}>
          <Button
            variant="outlined"
            color="error"
            fullWidth
            onClick={onDiscard}
            disabled={saved}
          >
            Discard
          </Button>
        </Grid>
      </Grid>
      {saved && (
        <Alert severity="success" sx={{ mt: 2 }}>
          v7 model saved successfully!
        </Alert>
      )}
    </Paper>
  );
}

// -- Main page ----------------------------------------------------------------

export default function UploadPage() {
  const queryClient = useQueryClient();

  // Upload state
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<number | "new">("new");
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Fit state
  const [holdTypeOverrides, setHoldTypeOverrides] = useState<Map<number, string>>(new Map());
  const [fitError, setFitError] = useState<string | null>(null);
  const [fitResult, setFitResult] = useState<V7FitPreviewResponse | null>(null);
  const [fitNotes, setFitNotes] = useState("");
  const [saved, setSaved] = useState(false);
  const [fitting, setFitting] = useState(false);
  const [fitProgress, setFitProgress] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Existing sessions
  const { data: sessions } = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
  });

  const [searchParams, setSearchParams] = useSearchParams();

  const { data: existingSession } = useQuery({
    queryKey: ["session", selectedSessionId],
    queryFn: () => getSession(selectedSessionId as number),
    enabled: typeof selectedSessionId === "number",
  });

  const activeSession = session ?? existingSession ?? null;

  // Load saved model predictions for existing sessions
  const [savedPredictions, setSavedPredictions] = useState<Map<number, V7FitPrediction>>(new Map());
  const { data: allModels } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
    enabled: !!activeSession,
  });

  useEffect(() => {
    if (!activeSession || !allModels) return;
    const sessionHoldIds = new Set(activeSession.holds.map((h) => h.id));

    const fetchPredictions = async () => {
      const predMap = new Map<number, V7FitPrediction>();
      const activeModel = allModels.versions.find((v) => v.is_active);
      if (!activeModel) return;
      if (!activeModel.hold_ids.some((id) => sessionHoldIds.has(id))) return;

      try {
        const predictions = await getHoldPredictions(activeModel.id);
        for (const pred of predictions) {
          if (sessionHoldIds.has(pred.hold_id)) {
            predMap.set(pred.hold_id, pred);
          }
        }
      } catch {
        // skip
      }
      setSavedPredictions(predMap);
    };

    fetchPredictions();
  }, [activeSession?.id, allModels]);

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: uploadSession,
    onSuccess: (data) => {
      setSession(data);
      setSelectedSessionId("new");
      setUploadError(null);
      setFitResult(null);
      setHoldTypeOverrides(new Map());
      setSaved(false);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
    onError: (err: { response?: { data?: { detail?: string } } }) => {
      setUploadError(err.response?.data?.detail ?? "Upload failed");
    },
  });

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadMutation.mutate(acceptedFiles[0]);
      }
    },
    [uploadMutation],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
  });

  const getHoldType = useCallback(
    (hold: HoldSummary) => holdTypeOverrides.get(hold.id) ?? hold.hold_type,
    [holdTypeOverrides],
  );

  // Tagged (non-untagged) holds for the fit
  const taggedHoldIds = useMemo(() => {
    if (!activeSession) return [];
    return activeSession.holds
      .filter((h) => getHoldType(h) !== "untagged")
      .map((h) => h.id);
  }, [activeSession, getHoldType]);

  // Build prediction lookup
  const predictionMap = useMemo(() => {
    const map = new Map<number, V7FitPrediction>(savedPredictions);
    if (fitResult) {
      for (const p of fitResult.predictions) {
        map.set(p.hold_id, p);
      }
    }
    return map;
  }, [savedPredictions, fitResult]);

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: saveFit,
    onSuccess: () => {
      setSaved(true);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Run v7 fit with task polling
  const handleRunFit = async () => {
    if (taggedHoldIds.length === 0) return;
    setFitError(null);
    setFitting(true);
    setFitResult(null);
    setSaved(false);
    setFitProgress("Submitting fit...");

    try {
      const { task_id } = await submitFitPreview({ hold_ids: taggedHoldIds });
      setFitProgress("Fitting (this may take a few minutes)...");

      // Poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const status = await pollFitTask(task_id);
          if (status.status === "completed" && status.result) {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setFitResult(status.result);
            setFitting(false);
            setFitProgress(null);
          } else if (status.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setFitError(status.error ?? "Fit failed");
            setFitting(false);
            setFitProgress(null);
          }
        } catch (err: any) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setFitError(err.response?.data?.detail ?? "Polling failed");
          setFitting(false);
          setFitProgress(null);
        }
      }, 3000);
    } catch (err: any) {
      setFitError(err.response?.data?.detail ?? "Failed to submit fit");
      setFitting(false);
      setFitProgress(null);
    }
  };

  const handleSave = () => {
    if (!fitResult) return;
    saveMutation.mutate({
      stage_a: fitResult.stage_a,
      stage_b: fitResult.stage_b,
      predictions: fitResult.predictions,
      overall_r2: fitResult.overall_r2,
      hold_ids: fitResult.hold_ids,
      s_base_values: {},  // Server-side already has these
      frozen_sensor: {},  // Server-side already has these
      notes: fitNotes || undefined,
      set_active: true,
    });
  };

  const handleDiscard = () => {
    setFitResult(null);
    setFitNotes("");
  };

  const handleSelectExistingSession = useCallback((id: number | "new") => {
    setSelectedSessionId(id);
    setSession(null);
    setFitResult(null);
    setHoldTypeOverrides(new Map());
    setSaved(false);
    setSavedPredictions(new Map());
  }, []);

  useEffect(() => {
    const sessionParam = searchParams.get("session");
    if (sessionParam && sessions) {
      const id = parseInt(sessionParam, 10);
      if (!isNaN(id) && sessions.some((s) => s.id === id)) {
        handleSelectExistingSession(id);
        setSearchParams({}, { replace: true });
      }
    }
  }, [searchParams, sessions, handleSelectExistingSession, setSearchParams]);

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      setSession(null);
      setSelectedSessionId("new");
      setFitResult(null);
      setHoldTypeOverrides(new Map());
      setSaved(false);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  const handleDeleteSession = () => {
    const id = activeSession?.id;
    if (!id) return;
    if (!window.confirm(
      "Delete this session and all its holds? You can re-upload the same CSV to re-import it."
    )) return;
    deleteMutation.mutate(id);
  };

  const taggedCount = taggedHoldIds.length;

  return (
    <Box sx={{ pb: activeSession ? 10 : 0 }}>
      <Typography variant="h4" gutterBottom>
        Upload & Fit
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload a CSV, tag holds, and fit the v7 two-stage model across all tagged holds simultaneously.
      </Typography>

      {/* Section 1: Upload / Select Session */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2, display: "flex", alignItems: "center", gap: 1 }}>
          <UploadIcon fontSize="small" />
          Session
        </Typography>

        <Grid container spacing={2} alignItems="stretch">
          <Grid size={{ xs: 12, md: 4 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Load Existing Session</InputLabel>
              <Select
                value={selectedSessionId}
                label="Load Existing Session"
                onChange={(e) => handleSelectExistingSession(e.target.value as number | "new")}
              >
                <MenuItem value="new">
                  <em>Upload New CSV</em>
                </MenuItem>
                {sessions?.map((s) => (
                  <MenuItem key={s.id} value={s.id}>
                    {s.name} ({s.session_date}) - {s.n_holds} holds
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid size={{ xs: 12, md: 8 }}>
            <Paper
              {...getRootProps()}
              sx={{
                p: 3,
                textAlign: "center",
                cursor: "pointer",
                border: "2px dashed",
                borderColor: isDragActive ? "primary.main" : "divider",
                bgcolor: isDragActive ? "rgba(37,99,235,0.04)" : "transparent",
                transition: "all 0.2s",
                "&:hover": {
                  borderColor: "primary.main",
                  bgcolor: "rgba(37,99,235,0.02)",
                },
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <input {...getInputProps()} />
              {uploadMutation.isPending ? (
                <CircularProgress />
              ) : (
                <>
                  <UploadIcon sx={{ fontSize: 36, color: "text.secondary", mb: 0.5 }} />
                  <Typography variant="body2" color="text.secondary">
                    {isDragActive ? "Drop your CSV here" : "Drag & drop CSV, or click to browse"}
                  </Typography>
                </>
              )}
            </Paper>
          </Grid>
        </Grid>

        {uploadError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {uploadError}
          </Alert>
        )}
      </Paper>

      {/* Section 2: Holds */}
      {activeSession && (
        <>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Box>
                <Typography variant="h6" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <FitIcon fontSize="small" />
                  Holds - {activeSession.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Tag holds with their type (FRC/RV/FL). All tagged holds are fitted together.
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Chip label={activeSession.session_date} size="small" />
                <Chip label={`${activeSession.holds.length} holds`} size="small" color="primary" />
                <Tooltip title="Delete this session and re-upload the CSV.">
                  <Button
                    size="small"
                    color="error"
                    variant="outlined"
                    startIcon={<DeleteIcon />}
                    onClick={handleDeleteSession}
                    disabled={deleteMutation.isPending}
                  >
                    {deleteMutation.isPending ? "Deleting..." : "Delete Session"}
                  </Button>
                </Tooltip>
              </Box>
            </Box>

            {fitError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {fitError}
              </Alert>
            )}

            {fitting && fitProgress && (
              <Box sx={{ mb: 2 }}>
                <Alert severity="info" sx={{ mb: 1 }}>
                  {fitProgress}
                </Alert>
                <LinearProgress />
              </Box>
            )}

            {activeSession.holds.map((hold) => (
              <HoldCard
                key={hold.id}
                hold={hold}
                prediction={predictionMap.get(hold.id)}
                onTagChange={(holdId, newType) => {
                  setHoldTypeOverrides((prev) => new Map(prev).set(holdId, newType));
                }}
              />
            ))}
          </Paper>

          {/* Section 3: Fit Results */}
          {fitResult && (
            <V7FitResultCard
              result={fitResult}
              notes={fitNotes}
              onNotesChange={setFitNotes}
              onSave={handleSave}
              onDiscard={handleDiscard}
              saving={saveMutation.isPending}
              saved={saved}
            />
          )}
        </>
      )}

      {/* Fixed bottom fit action bar */}
      {activeSession && (
        <Paper
          elevation={4}
          sx={{
            position: "fixed",
            bottom: 0,
            left: { xs: 0, md: 240 },
            right: 0,
            zIndex: 20,
            display: "flex",
            alignItems: "center",
            gap: 2,
            px: 3,
            py: 1.5,
            borderTop: "1px solid",
            borderColor: "divider",
            borderRadius: 0,
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
            {taggedCount === 0
              ? "No holds tagged - tag holds to include them in the fit"
              : `${taggedCount} hold${taggedCount !== 1 ? "s" : ""} tagged for fitting`}
          </Typography>

          <Button
            variant="contained"
            onClick={handleRunFit}
            disabled={taggedCount === 0 || fitting}
            startIcon={fitting ? <CircularProgress size={18} /> : <FitIcon />}
            sx={{ minWidth: 120 }}
          >
            {fitting ? "Fitting..." : "Fit Model"}
          </Button>
        </Paper>
      )}
    </Box>
  );
}
