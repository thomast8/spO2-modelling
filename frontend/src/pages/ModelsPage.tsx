import {
  CheckCircle,
  ExpandLess,
  ExpandMore,
  RadioButtonUnchecked,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import "katex/dist/katex.min.css";
import { useState } from "react";
import { InlineMath } from "react-katex";
import { Link as RouterLink } from "react-router-dom";
import { activateModel, getPredictionCurve, listAllModels } from "../api/models";
import type { V7ModelVersionResponse } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";
import { MODEL_SUMMARY } from "../constants/modelDescriptions";

function ModelDetail({ model }: { model: V7ModelVersionResponse }) {
  const [selectedHoldId, setSelectedHoldId] = useState<number>(
    model.hold_ids[0] ?? 0,
  );

  const { data: curve } = useQuery({
    queryKey: ["prediction", model.id, selectedHoldId],
    queryFn: () => getPredictionCurve(model.id, selectedHoldId),
    enabled: selectedHoldId > 0,
  });

  return (
    <Box sx={{ mt: 2, mb: 2 }}>
      {/* Stage A params */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: "text.secondary" }}>
        Stage A: Sensor (tau_0 = {model.stage_a.tau_0.toFixed(2)}s, p = {model.stage_a.p.toFixed(2)})
      </Typography>
      <Table size="small" sx={{ mb: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell>Hold</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>delta</TableCell>
            <TableCell>S_min</TableCell>
            <TableCell>Eff. Lag</TableCell>
            <TableCell>R\u00B2</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {model.stage_a.per_hold.map((ph) => (
            <TableRow key={ph.hold_id}>
              <TableCell>#{ph.hold_id}</TableCell>
              <TableCell>{ph.hold_type}</TableCell>
              <TableCell sx={{ fontFamily: "monospace" }}>{ph.delta.toFixed(2)}</TableCell>
              <TableCell sx={{ fontFamily: "monospace" }}>{ph.s_min.toFixed(1)}</TableCell>
              <TableCell sx={{ fontFamily: "monospace" }}>{ph.eff_lag.toFixed(1)}</TableCell>
              <TableCell sx={{ fontFamily: "monospace" }}>{ph.r_squared.toFixed(4)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {/* Stage B params */}
      <Typography variant="subtitle2" sx={{ mb: 1, color: "text.secondary" }}>
        Stage B: Physiology (k_co2 = {model.stage_b.k_co2.toFixed(4)}, gamma = {model.stage_b.gamma.toFixed(2)})
      </Typography>
      <Table size="small" sx={{ mb: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell>Hold</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>tau_washout</TableCell>
            <TableCell>paco2_0</TableCell>
            <TableCell>R\u00B2</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {model.stage_b.per_hold.map((ph) => (
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

      {/* Prediction curve per hold */}
      <Box sx={{ display: "flex", gap: 2, alignItems: "center", mb: 1 }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Hold</InputLabel>
          <Select
            value={selectedHoldId}
            label="Hold"
            onChange={(e) => setSelectedHoldId(Number(e.target.value))}
          >
            {model.hold_ids.map((id) => (
              <MenuItem key={id} value={id}>Hold #{id}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      {curve && (
        <SpO2Chart
          predictedT={curve.t}
          predictedSpo2={curve.spo2}
          title={`v${model.version} - Hold #${selectedHoldId}`}
          height={280}
        />
      )}
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
            <Chip label="v7 Two-Stage" size="small" color="primary" variant="outlined" />
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

  const versions = models?.versions ?? [];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Versions
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        View, compare, and activate v7 model versions. Each model fits all hold types simultaneously.
      </Typography>

      <ModelSummaryCard />

      {versions.length === 0 ? (
        <Alert severity="info">
          No v7 models yet. Go to the Upload & Fit page to create one.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Version</TableCell>
                <TableCell>R\u00B2</TableCell>
                <TableCell>Holds</TableCell>
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
                    <TableCell>{v.hold_ids.length}</TableCell>
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
