import { CheckCircle, RadioButtonUnchecked } from "@mui/icons-material";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  IconButton,
  Paper,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { activateModel, getPredictionCurve, listAllModels } from "../api/models";
import type { HoldType, ModelVersionResponse } from "../api/types";
import SpO2Chart from "../components/charts/SpO2Chart";

const HOLD_TYPES: HoldType[] = ["FRC", "RV", "FL"];

function ModelDetail({ model }: { model: ModelVersionResponse }) {
  const { data: curve } = useQuery({
    queryKey: ["prediction", model.id],
    queryFn: () => getPredictionCurve(model.id),
  });

  return (
    <Box sx={{ mt: 2 }}>
      {curve && (
        <SpO2Chart
          predictedT={curve.t}
          predictedSpo2={curve.spo2}
          title={`v${model.version} Prediction Curve`}
          height={300}
        />
      )}
    </Box>
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
                        {v.notes ?? "—"}
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
