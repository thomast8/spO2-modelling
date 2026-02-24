import {
  ChevronRight as ChevronRightIcon,
  CloudUpload as UploadIcon,
  FolderOpen as SessionIcon,
  ModelTraining as ModelIcon,
} from "@mui/icons-material";
import { Alert, Box, CircularProgress, Grid, Paper, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { listAllModels } from "../api/models";
import { useNavigate } from "react-router-dom";
import { listSessions } from "../api/sessions";
import StatCard from "../components/StatCard";

export default function DashboardPage() {
  const navigate = useNavigate();
  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
  });

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
  });

  if (sessionsLoading || modelsLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  const totalHolds = sessions?.reduce((acc, s) => acc + s.n_holds, 0) ?? 0;
  const totalVersions =
    (models?.FRC.versions.length ?? 0) +
    (models?.RV.versions.length ?? 0) +
    (models?.FL.versions.length ?? 0);

  const activeModels = [
    models?.FRC.active_version ? "FRC" : null,
    models?.RV.active_version ? "RV" : null,
    models?.FL.active_version ? "FL" : null,
  ].filter(Boolean);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        SpO₂ desaturation modelling — Hill equation ODC fits for breath-hold apnea
      </Typography>

      <Grid container spacing={2.5} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            label="Sessions"
            value={sessions?.length ?? 0}
            icon={<SessionIcon />}
            color="primary.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            label="Total Holds"
            value={totalHolds}
            icon={<UploadIcon />}
            color="secondary.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            label="Model Versions"
            value={totalVersions}
            icon={<ModelIcon />}
            color="success.main"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            label="Active Models"
            value={activeModels.length > 0 ? activeModels.join(", ") : "None"}
            icon={<ModelIcon />}
            color="warning.main"
          />
        </Grid>
      </Grid>

      {sessions?.length === 0 && (
        <Paper sx={{ p: 4, textAlign: "center" }}>
          <Alert severity="info" sx={{ justifyContent: "center" }}>
            No sessions yet. Upload a CSV file to get started!
          </Alert>
        </Paper>
      )}

      {sessions && sessions.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recent Sessions
          </Typography>
          {sessions.slice(0, 5).map((s) => (
            <Box
              key={s.id}
              onClick={() => navigate(`/upload?session=${s.id}`)}
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                py: 1.5,
                px: 1.5,
                borderBottom: "1px solid",
                borderColor: "divider",
                "&:last-child": { borderBottom: "none" },
                cursor: "pointer",
                borderRadius: 1,
                transition: "background-color 0.15s",
                "&:hover": { bgcolor: "rgba(37, 99, 235, 0.04)" },
              }}
            >
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  {s.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {s.csv_filename} — {s.session_date}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  {s.n_holds} holds
                </Typography>
                <ChevronRightIcon sx={{ color: "text.secondary", fontSize: 20 }} />
              </Box>
            </Box>
          ))}
        </Paper>
      )}
    </Box>
  );
}
