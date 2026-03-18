import api from "./client";
import type {
  V7FitPrediction,
  V7ModelVersionListResponse,
  V7ModelVersionResponse,
  V7PredictionCurve,
} from "./types";

export async function listAllModels(): Promise<V7ModelVersionListResponse> {
  const { data } = await api.get<V7ModelVersionListResponse>("/models");
  return data;
}

export async function getActiveModel(): Promise<V7ModelVersionResponse> {
  const { data } = await api.get<V7ModelVersionResponse>("/models/active");
  return data;
}

export async function activateModel(
  modelId: number,
): Promise<V7ModelVersionResponse> {
  const { data } = await api.post<V7ModelVersionResponse>(
    `/models/${modelId}/activate`,
  );
  return data;
}

export async function getPredictionCurve(
  modelId: number,
  holdId: number,
  tMax = 600,
  dt = 1,
): Promise<V7PredictionCurve> {
  const { data } = await api.get<V7PredictionCurve>(
    `/models/${modelId}/predict`,
    { params: { hold_id: holdId, t_max: tMax, dt } },
  );
  return data;
}

export async function getHoldPredictions(
  modelId: number,
): Promise<V7FitPrediction[]> {
  const { data } = await api.get<V7FitPrediction[]>(
    `/models/${modelId}/hold-predictions`,
  );
  return data;
}
