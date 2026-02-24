import api from "./client";
import type { SessionListItem, SessionResponse } from "./types";

export async function uploadSession(file: File): Promise<SessionResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<SessionResponse>("/sessions/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listSessions(): Promise<SessionListItem[]> {
  const { data } = await api.get<SessionListItem[]>("/sessions");
  return data;
}

export async function getSession(id: number): Promise<SessionResponse> {
  const { data } = await api.get<SessionResponse>(`/sessions/${id}`);
  return data;
}

export async function deleteSession(id: number): Promise<void> {
  await api.delete(`/sessions/${id}`);
}
