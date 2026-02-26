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
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useSearchParams } from "react-router-dom";
import { previewFit, saveFit } from "../api/fit";
import { getHold, updateHold } from "../api/holds";
import { getHoldPredictions, listAllModels } from "../api/models";
import { deleteSession, listSessions, getSession, uploadSession } from "../api/sessions";
import type {
  FitPrediction,
  FitPreviewResponse,
  HoldSummary,
  HoldType,
  SessionResponse,
} from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { PARAM_DESCRIPTIONS } from "../constants/modelDescriptions";

const HOLD_TYPE_OPTIONS = ["untagged", "FRC", "RV", "FL"] as const;
const HOLD_TYPE_LABELS: Record<string, string> = { untagged: "Skip" };
const FIT_HOLD_TYPES: HoldType[] = ["FRC", "RV", "FL"];

const PARAM_LABELS: Record<string, string> = {
  pao2_0: "PAO\u2082 Initial (mmHg)",
  pvo2: "PvO\u2082 (mmHg)",
  tau_washout: "\u03C4 Washout (s)",
  bohr_max: "Bohr Max \u0394P50 (mmHg)",
  tau_bohr: "\u03C4 Bohr (s)",
  r_offset: "Offset (%)",
  p50_base: "P50 Base (fixed)",
  gamma: "Steepness (\u03B3)",
};

// ── Hold card with optional fit overlay ──────────────────────

function HoldCard({
  hold,
  prediction,
  onTagChange,
}: {
  hold: HoldSummary;
  prediction?: FitPrediction;
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

// ── Fit result card per hold type ────────────────────────────

function FitResultCard({
  holdType,
  result,
  notes,
  onNotesChange,
  onSave,
  onDiscard,
  saving,
  saved,
}: {
  holdType: HoldType;
  result: FitPreviewResponse;
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
        <Typography variant="h6">
          {holdType} Fit Results
        </Typography>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Chip
            label={`R\u00B2 = ${result.r_squared.toFixed(4)}`}
            color={result.r_squared > 0.95 ? "success" : result.r_squared > 0.9 ? "warning" : "error"}
          />
          <Chip
            label={result.converged ? "Converged" : "Not converged"}
            color={result.converged ? "success" : "warning"}
            variant="outlined"
          />
          <Chip label={`${result.n_holds} hold${result.n_holds !== 1 ? "s" : ""}`} size="small" variant="outlined" />
        </Box>
      </Box>

      <TableContainer sx={{ mb: 3 }}>
        <Table size="small">
          <TableBody>
            {Object.entries(result.params).map(([key, val]) => (
              <TableRow key={key}>
                <TableCell sx={{ fontWeight: 600, border: "none", py: 0.5 }}>
                  <Tooltip title={PARAM_DESCRIPTIONS[key] ?? ""} arrow placement="right">
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, cursor: "help" }}>
                      {PARAM_LABELS[key] ?? key}
                      {PARAM_DESCRIPTIONS[key] && (
                        <InfoIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                      )}
                    </Box>
                  </Tooltip>
                </TableCell>
                <TableCell sx={{ border: "none", fontFamily: "monospace", py: 0.5 }}>
                  {typeof val === "number" ? val.toFixed(4) : val}
                </TableCell>
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
            placeholder={`e.g., ${holdType} model from session data`}
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
            {saving ? <CircularProgress size={22} /> : saved ? "Saved" : `Save ${holdType}`}
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
          {holdType} model saved successfully!
        </Alert>
      )}
    </Paper>
  );
}

// ── Main page ────────────────────────────────────────────────

export default function UploadPage() {
  const queryClient = useQueryClient();

  // Upload state
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<number | "new">("new");
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Fit state — selected hold IDs (across all types)
  const [selectedHoldIds, setSelectedHoldIds] = useState<number[]>([]);
  // Local hold type overrides (tracks user tag changes that may not yet be in activeSession)
  const [holdTypeOverrides, setHoldTypeOverrides] = useState<Map<number, string>>(new Map());
  const [fitError, setFitError] = useState<string | null>(null);
  // Results keyed by hold type
  const [fitResults, setFitResults] = useState<Map<HoldType, FitPreviewResponse>>(new Map());
  const [fitNotes, setFitNotes] = useState<Map<HoldType, string>>(new Map());
  const [savedTypes, setSavedTypes] = useState<Set<HoldType>>(new Set());
  const [fitting, setFitting] = useState(false);

  // Existing sessions
  const { data: sessions } = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
  });

  const [searchParams, setSearchParams] = useSearchParams();

  // Load existing session detail
  const { data: existingSession } = useQuery({
    queryKey: ["session", selectedSessionId],
    queryFn: () => getSession(selectedSessionId as number),
    enabled: typeof selectedSessionId === "number",
  });

  const activeSession = session ?? existingSession ?? null;

  // Load saved model predictions for existing sessions
  const [savedPredictions, setSavedPredictions] = useState<Map<number, FitPrediction>>(new Map());
  const { data: allModels } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
    enabled: !!activeSession,
  });

  useEffect(() => {
    if (!activeSession || !allModels) return;
    const sessionHoldIds = new Set(activeSession.holds.map((h) => h.id));

    const fetchPredictions = async () => {
      const predMap = new Map<number, FitPrediction>();
      for (const holdType of ["FRC", "RV", "FL"] as const) {
        const typeData = allModels[holdType];
        const activeModel = typeData.versions.find((v) => v.is_active);
        if (!activeModel) continue;
        const modelHoldIds = activeModel.hold_ids;
        if (!modelHoldIds.some((id) => sessionHoldIds.has(id))) continue;

        try {
          const predictions = await getHoldPredictions(activeModel.id);
          for (const pred of predictions) {
            if (sessionHoldIds.has(pred.hold_id)) {
              predMap.set(pred.hold_id, pred);
            }
          }
        } catch {
          // Model predictions unavailable — skip silently
        }
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
      setFitResults(new Map());
      setSelectedHoldIds([]);
      setHoldTypeOverrides(new Map());
      setSavedTypes(new Set());
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

  // Resolve hold type: local override takes priority over session data
  const getHoldType = useCallback(
    (hold: HoldSummary) => holdTypeOverrides.get(hold.id) ?? hold.hold_type,
    [holdTypeOverrides],
  );

  // Group selected holds by type for the fit
  const selectedByType = useMemo(() => {
    if (!activeSession) return new Map<HoldType, number[]>();
    const map = new Map<HoldType, number[]>();
    for (const hold of activeSession.holds) {
      const type = getHoldType(hold);
      if (selectedHoldIds.includes(hold.id) && type !== "untagged") {
        const ids = map.get(type as HoldType) ?? [];
        ids.push(hold.id);
        map.set(type as HoldType, ids);
      }
    }
    return map;
  }, [activeSession, selectedHoldIds, getHoldType]);

  // Build prediction lookup: saved model predictions as base, fresh fit results on top
  const predictionMap = useMemo(() => {
    const map = new Map<number, FitPrediction>(savedPredictions);
    for (const result of fitResults.values()) {
      for (const p of result.predictions) {
        map.set(p.hold_id, p);
      }
    }
    return map;
  }, [savedPredictions, fitResults]);

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: saveFit,
    onSuccess: (_data, variables) => {
      setSavedTypes((prev) => new Set(prev).add(variables.hold_type as HoldType));
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  // Run fit for all selected types in parallel
  const handleRunFit = async () => {
    if (selectedByType.size === 0) return;
    setFitError(null);
    setFitting(true);
    setFitResults(new Map());
    setSavedTypes(new Set());

    const newResults = new Map<HoldType, FitPreviewResponse>();
    const errors: string[] = [];

    await Promise.all(
      Array.from(selectedByType.entries()).map(async ([holdType, holdIds]) => {
        try {
          const result = await previewFit({ hold_type: holdType, hold_ids: holdIds });
          newResults.set(holdType, result);
        } catch (err: any) {
          errors.push(`${holdType}: ${err.response?.data?.detail ?? "Fit failed"}`);
        }
      }),
    );

    setFitResults(newResults);
    if (errors.length > 0) setFitError(errors.join("; "));
    setFitting(false);
  };

  const handleSave = (holdType: HoldType) => {
    const result = fitResults.get(holdType);
    if (!result) return;
    const holdIds = selectedByType.get(holdType) ?? [];
    saveMutation.mutate({
      hold_type: holdType,
      params: result.params,
      hold_ids: holdIds,
      r_squared: result.r_squared,
      objective_val: result.objective_val,
      converged: result.converged,
      notes: fitNotes.get(holdType) || undefined,
      set_active: true,
    });
  };

  const handleDiscard = (holdType: HoldType) => {
    setFitResults((prev) => {
      const next = new Map(prev);
      next.delete(holdType);
      return next;
    });
  };

  const handleSaveAll = () => {
    for (const holdType of fitResults.keys()) {
      if (!savedTypes.has(holdType)) handleSave(holdType);
    }
  };

  const handleSelectExistingSession = useCallback((id: number | "new") => {
    setSelectedSessionId(id);
    setSession(null);
    setFitResults(new Map());
    setSelectedHoldIds([]);
    setHoldTypeOverrides(new Map());
    setSavedTypes(new Set());
    setSavedPredictions(new Map());
  }, []);

  // Auto-select session from URL query param (e.g., /upload?session=123)
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

  // Delete session mutation
  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      setSession(null);
      setSelectedSessionId("new");
      setFitResults(new Map());
      setSelectedHoldIds([]);
      setHoldTypeOverrides(new Map());
      setSavedTypes(new Set());
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

  // Summary of selected types
  const selectedTypeSummary = useMemo(() => {
    return Array.from(selectedByType.entries())
      .map(([type, ids]) => `${ids.length} ${type}`)
      .join(", ");
  }, [selectedByType]);

  return (
    <Box sx={{ pb: activeSession ? 10 : 0 }}>
      <Typography variant="h4" gutterBottom>
        Upload & Fit
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload a CSV, tag holds, select them, and fit models for all selected types at once.
      </Typography>

      {/* ── Section 1: Upload / Select Session ─────────────────── */}
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
                    {s.name} ({s.session_date}) — {s.n_holds} holds
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

      {/* ── Section 2: Holds — Tag & Select ────────────────────── */}
      {activeSession && (
        <>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Box>
                <Typography variant="h6" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <FitIcon fontSize="small" />
                  Holds — {activeSession.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Tag holds to select them for fitting. Fit all selected types at once.
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Chip label={activeSession.session_date} size="small" />
                <Chip label={`${activeSession.holds.length} holds`} size="small" color="primary" />
                <Tooltip title="Delete this session and re-upload the CSV. Useful when data processing has been updated and you want to re-import with the latest corrections applied.">
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

            {/* Hold cards */}
            {activeSession.holds.map((hold) => (
                <HoldCard
                  key={hold.id}
                  hold={hold}
                  prediction={predictionMap.get(hold.id)}
                  onTagChange={(holdId, newType) => {
                    // Track the type change locally for immediate reactivity
                    setHoldTypeOverrides((prev) => new Map(prev).set(holdId, newType));
                    if (newType !== "untagged") {
                      // Auto-select when tagged
                      setSelectedHoldIds((prev) =>
                        prev.includes(holdId) ? prev : [...prev, holdId],
                      );
                    } else {
                      // Auto-deselect when untagged
                      setSelectedHoldIds((prev) => prev.filter((id) => id !== holdId));
                    }
                  }}
                />
            ))}
          </Paper>

          {/* ── Section 3: Fit Results & Save ──────────────────── */}
          {fitResults.size > 0 && (
            <>
              {fitResults.size > 1 && (
                <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
                  <Button
                    variant="contained"
                    color="success"
                    onClick={handleSaveAll}
                    disabled={savedTypes.size === fitResults.size || saveMutation.isPending}
                  >
                    Save All Models
                  </Button>
                </Box>
              )}
              {FIT_HOLD_TYPES.filter((t) => fitResults.has(t)).map((holdType) => (
                <FitResultCard
                  key={holdType}
                  holdType={holdType}
                  result={fitResults.get(holdType)!}
                  notes={fitNotes.get(holdType) ?? ""}
                  onNotesChange={(v) => setFitNotes((prev) => new Map(prev).set(holdType, v))}
                  onSave={() => handleSave(holdType)}
                  onDiscard={() => handleDiscard(holdType)}
                  saving={saveMutation.isPending && saveMutation.variables?.hold_type === holdType}
                  saved={savedTypes.has(holdType)}
                />
              ))}
            </>
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
            {selectedHoldIds.length === 0
              ? "No holds selected \u2014 tag holds to include them in the fit"
              : `Selected: ${selectedTypeSummary}`}
          </Typography>

          <Button
            variant="contained"
            onClick={handleRunFit}
            disabled={selectedByType.size === 0 || fitting}
            startIcon={fitting ? <CircularProgress size={18} /> : <FitIcon />}
            sx={{ minWidth: 120 }}
          >
            {fitting ? "Fitting..." : `Fit ${selectedByType.size > 1 ? `${selectedByType.size} Types` : "Model"}`}
          </Button>
        </Paper>
      )}
    </Box>
  );
}
