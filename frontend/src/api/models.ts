import api from "./client";
import type {
  AllModelsResponse,
  ModelVersionListResponse,
  ModelVersionResponse,
  PredictionCurve,
} from "./types";

export async function listAllModels(): Promise<AllModelsResponse> {
  const { data } = await api.get<AllModelsResponse>("/models");
  return data;
}

export async function listModelsForType(holdType: string): Promise<ModelVersionListResponse> {
  const { data } = await api.get<ModelVersionListResponse>(`/models/${holdType}`);
  return data;
}

export async function getActiveModel(holdType: string): Promise<ModelVersionResponse> {
  const { data } = await api.get<ModelVersionResponse>(`/models/${holdType}/active`);
  return data;
}

export async function activateModel(modelId: number): Promise<ModelVersionResponse> {
  const { data } = await api.post<ModelVersionResponse>(`/models/${modelId}/activate`);
  return data;
}

export async function getPredictionCurve(
  modelId: number,
  tMax = 600,
  dt = 1,
): Promise<PredictionCurve> {
  const { data } = await api.get<PredictionCurve>(`/models/${modelId}/predict`, {
    params: { t_max: tMax, dt },
  });
  return data;
}
