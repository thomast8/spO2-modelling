import api from "./client";
import type { DesatRatePoint, SensitivityPoint, ThresholdResponse } from "./types";

export async function getThreshold(
  modelId: number,
  threshold = 40,
  tMax = 800,
): Promise<ThresholdResponse> {
  const { data } = await api.get<ThresholdResponse>("/analysis/threshold", {
    params: { model_id: modelId, threshold, t_max: tMax },
  });
  return data;
}

export async function getSensitivity(
  modelId: number,
  referenceTimeS = 372,
  threshold = 40,
  tMax = 800,
): Promise<SensitivityPoint[]> {
  const { data } = await api.get<SensitivityPoint[]>("/analysis/sensitivity", {
    params: { model_id: modelId, reference_time_s: referenceTimeS, threshold, t_max: tMax },
  });
  return data;
}

export async function getDesatRate(
  modelId: number,
  timePoints = "60,120,180,240,300",
  tMax = 800,
): Promise<DesatRatePoint[]> {
  const { data } = await api.get<DesatRatePoint[]>("/analysis/desat-rate", {
    params: { model_id: modelId, time_points: timePoints, t_max: tMax },
  });
  return data;
}
