import api from "./client";
import type {
  TaskStatusResponse,
  V7FitPreviewRequest,
  V7FitSaveRequest,
  V7ModelVersionResponse,
} from "./types";

export async function submitFitPreview(
  request: V7FitPreviewRequest,
): Promise<{ task_id: string }> {
  const { data } = await api.post<{ task_id: string }>("/fit/preview", request);
  return data;
}

export async function pollFitTask(taskId: string): Promise<TaskStatusResponse> {
  const { data } = await api.get<TaskStatusResponse>(`/fit/task/${taskId}`);
  return data;
}

export async function saveFit(
  request: V7FitSaveRequest,
): Promise<V7ModelVersionResponse> {
  const { data } = await api.post<V7ModelVersionResponse>("/fit/save", request);
  return data;
}
