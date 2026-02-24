import api from "./client";
import type { FitPreviewRequest, FitPreviewResponse, FitSaveRequest, ModelVersionResponse } from "./types";

export async function previewFit(request: FitPreviewRequest): Promise<FitPreviewResponse> {
  const { data } = await api.post<FitPreviewResponse>("/fit/preview", request);
  return data;
}

export async function saveFit(request: FitSaveRequest): Promise<ModelVersionResponse> {
  const { data } = await api.post<ModelVersionResponse>("/fit/save", request);
  return data;
}
