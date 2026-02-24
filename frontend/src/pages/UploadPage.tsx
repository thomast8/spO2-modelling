import { CloudUpload as UploadIcon } from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControlLabel,
  MenuItem,
  Paper,
  Select,
  Switch,
  Typography,
} from "@mui/material";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { updateHold } from "../api/holds";
import { uploadSession } from "../api/sessions";
import type { HoldSummary, SessionResponse } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { getHold } from "../api/holds";
import { useQuery } from "@tanstack/react-query";

const HOLD_TYPES = ["untagged", "FRC", "RV", "FL"] as const;

function HoldCard({ hold }: { hold: HoldSummary }) {
  const queryClient = useQueryClient();

  const { data: holdDetail } = useQuery({
    queryKey: ["hold", hold.id],
    queryFn: () => getHold(hold.id),
  });

  const updateMutation = useMutation({
    mutationFn: (update: { hold_type?: string; include_in_fit?: boolean }) =>
      updateHold(hold.id, update),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["hold", hold.id] });
    },
  });

  const currentType = holdDetail?.hold_type ?? hold.hold_type;
  const currentInclude = holdDetail?.include_in_fit ?? hold.include_in_fit;

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="h6" sx={{ fontSize: "1rem" }}>
            Hold {hold.hold_number}
          </Typography>
          <Chip
            label={`${hold.duration_s}s`}
            size="small"
            color="primary"
            variant="outlined"
          />
          {hold.min_spo2 !== null && (
            <Chip
              label={`Min SpO₂: ${hold.min_spo2}%`}
              size="small"
              color={hold.min_spo2 < 60 ? "error" : hold.min_spo2 < 80 ? "warning" : "default"}
              variant="outlined"
            />
          )}
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Select
            value={currentType}
            onChange={(e) => updateMutation.mutate({ hold_type: e.target.value })}
            size="small"
            sx={{ minWidth: 100 }}
          >
            {HOLD_TYPES.map((t) => (
              <MenuItem key={t} value={t}>
                {t}
              </MenuItem>
            ))}
          </Select>
          <FormControlLabel
            control={
              <Switch
                checked={currentInclude}
                onChange={(e) => updateMutation.mutate({ include_in_fit: e.target.checked })}
                size="small"
              />
            }
            label="Fit"
            sx={{ "& .MuiTypography-root": { fontSize: "0.8rem" } }}
          />
        </Box>
      </Box>
      {holdDetail?.data_points && (
        <SpO2Chart
          observedT={holdDetail.data_points.map((dp) => dp.elapsed_s)}
          observedSpo2={holdDetail.data_points.map((dp) => dp.spo2)}
          height={200}
        />
      )}
    </Paper>
  );
}

export default function UploadPage() {
  const queryClient = useQueryClient();
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const uploadMutation = useMutation({
    mutationFn: uploadSession,
    onSuccess: (data) => {
      setSession(data);
      setError(null);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
    onError: (err: { response?: { data?: { detail?: string } } }) => {
      setError(err.response?.data?.detail ?? "Upload failed");
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

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Upload Session
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload a CSV file from your pulse oximeter to detect apnea holds.
      </Typography>

      {/* Dropzone */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 6,
          textAlign: "center",
          cursor: "pointer",
          border: "2px dashed",
          borderColor: isDragActive ? "primary.main" : "rgba(255,255,255,0.15)",
          bgcolor: isDragActive ? "rgba(79,195,247,0.05)" : "transparent",
          transition: "all 0.2s",
          "&:hover": {
            borderColor: "primary.main",
            bgcolor: "rgba(79,195,247,0.03)",
          },
          mb: 3,
        }}
      >
        <input {...getInputProps()} />
        {uploadMutation.isPending ? (
          <CircularProgress />
        ) : (
          <>
            <UploadIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
            <Typography variant="h6" color="text.secondary">
              {isDragActive ? "Drop your CSV here" : "Drag & drop a CSV file, or click to browse"}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Accepts .csv files from pulse oximeter sessions
            </Typography>
          </>
        )}
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Session result */}
      {session && (
        <Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            <Typography variant="h5">
              {session.name}
            </Typography>
            <Chip label={session.session_date} size="small" />
            <Chip label={`${session.holds.length} holds`} size="small" color="primary" />
          </Box>

          {session.holds.map((hold) => (
            <HoldCard key={hold.id} hold={hold} />
          ))}

          <Box sx={{ mt: 2, textAlign: "center" }}>
            <Button
              variant="outlined"
              onClick={() => setSession(null)}
            >
              Upload Another
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
}
