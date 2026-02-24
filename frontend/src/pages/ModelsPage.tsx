import {
  CheckCircle,
  ExpandLess,
  ExpandMore,
  InfoOutlined as InfoIcon,
  RadioButtonUnchecked,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  IconButton,
  Paper,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import "katex/dist/katex.min.css";
import { useState } from "react";
import { InlineMath } from "react-katex";
import { Link as RouterLink } from "react-router-dom";
import { activateModel, getPredictionCurve, listAllModels } from "../api/models";
import type { HoldType, ModelVersionResponse } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { MODEL_SUMMARY, PARAM_DESCRIPTIONS } from "../constants/modelDescriptions";

const HOLD_TYPES: HoldType[] = ["FRC", "RV", "FL"];

const PARAM_LABELS: Record<string, string> = {
  pao2_0: "PAO\u2082 Initial (mmHg)",
  pvo2: "PvO\u2082 (mmHg)",
  tau_washout: "\u03C4 Washout (s)",
  p50_base: "P50 Base (mmHg)",
  n: "Hill Coefficient",
  bohr_coeff: "Bohr Coeff (mmHg/s)",
  lag: "Lag (s)",
  r_offset: "Offset (%)",
};

function ModelDetail({ model }: { model: ModelVersionResponse }) {
  const { data: curve } = useQuery({
    queryKey: ["prediction", model.id],
    queryFn: () => getPredictionCurve(model.id),
  });

  return (
    <Box sx={{ mt: 2, mb: 2 }}>
      <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        {/* Parameter table with tooltips */}
        <Table size="small" sx={{ maxWidth: 400 }}>
          <TableBody>
            {Object.entries(model.params).map(([key, val]) => (
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

        {/* Prediction curve */}
        <Box sx={{ flex: 1, minWidth: 300 }}>
          {curve && (
            <SpO2Chart
              predictedT={curve.t}
              predictedSpo2={curve.spo2}
              title={`v${model.version} Prediction Curve`}
              height={280}
            />
          )}
        </Box>
      </Box>
    </Box>
  );
}

function ModelSummaryCard() {
  const [expanded, setExpanded] = useState(false);

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box
        sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", cursor: "pointer" }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {MODEL_SUMMARY.title}
            <Chip label="Hill ODC" size="small" color="primary" variant="outlined" />
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {MODEL_SUMMARY.description}
          </Typography>
        </Box>
        <IconButton size="small">
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box
          sx={{
            mt: 2,
            p: 2,
            bgcolor: "rgba(37, 99, 235, 0.04)",
            borderRadius: 2,
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
          }}
        >
          {MODEL_SUMMARY.equations.map((eq) => (
            <Box key={eq.label} sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
              <Typography
                component="span"
                sx={{ fontWeight: 600, color: "text.secondary", fontSize: "0.8rem", minWidth: 110 }}
              >
                {eq.label}:
              </Typography>
              <InlineMath math={eq.latex} />
            </Box>
          ))}
        </Box>
        <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end" }}>
          <Button
            component={RouterLink}
            to="/about-model"
            size="small"
            variant="text"
          >
            Learn more about the model
          </Button>
        </Box>
      </Collapse>
    </Paper>
  );
}

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState(0);
  const [expandedModel, setExpandedModel] = useState<number | null>(null);

  const { data: models, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: listAllModels,
  });

  const activateMutation = useMutation({
    mutationFn: activateModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  const holdType = HOLD_TYPES[activeTab];
  const typeData = models?.[holdType];
  const versions = typeData?.versions ?? [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Versions
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        View, compare, and activate model versions for each hold type.
      </Typography>

      <ModelSummaryCard />

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          sx={{
            "& .MuiTab-root": { fontWeight: 700, fontSize: "0.95rem" },
          }}
        >
          {HOLD_TYPES.map((t) => (
            <Tab
              key={t}
              label={
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {t}
                  <Chip
                    label={models?.[t].versions.length ?? 0}
                    size="small"
                    color="primary"
                    variant="outlined"
                    sx={{ height: 20, fontSize: "0.7rem" }}
                  />
                </Box>
              }
            />
          ))}
        </Tabs>
      </Paper>

      {versions.length === 0 ? (
        <Alert severity="info">
          No models for {holdType} yet. Go to the Fit page to create one.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Version</TableCell>
                <TableCell>R²</TableCell>
                <TableCell>Holds Used</TableCell>
                <TableCell>Converged</TableCell>
                <TableCell>Notes</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="center">Active</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {versions.map((v) => (
                <>
                  <TableRow
                    key={v.id}
                    hover
                    onClick={() => setExpandedModel(expandedModel === v.id ? null : v.id)}
                    sx={{ cursor: "pointer" }}
                  >
                    <TableCell>
                      <Typography fontWeight={700}>v{v.version}</Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={v.r_squared.toFixed(4)}
                        size="small"
                        color={v.r_squared > 0.95 ? "success" : v.r_squared > 0.9 ? "warning" : "error"}
                      />
                    </TableCell>
                    <TableCell>{v.n_holds_used}</TableCell>
                    <TableCell>{v.converged ? "Yes" : "No"}</TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                        {v.notes ?? "\u2014"}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(v.created_at).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <IconButton
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!v.is_active) activateMutation.mutate(v.id);
                        }}
                        color={v.is_active ? "success" : "default"}
                        size="small"
                      >
                        {v.is_active ? <CheckCircle /> : <RadioButtonUnchecked />}
                      </IconButton>
                    </TableCell>
                  </TableRow>
                  {expandedModel === v.id && (
                    <TableRow key={`${v.id}-detail`}>
                      <TableCell colSpan={7} sx={{ py: 0 }}>
                        <ModelDetail model={v} />
                      </TableCell>
                    </TableRow>
                  )}
                </>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}
