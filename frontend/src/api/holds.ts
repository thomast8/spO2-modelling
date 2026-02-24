import api from "./client";
import type { HoldDetailResponse } from "./types";

export async function getHold(id: number): Promise<HoldDetailResponse> {
  const { data } = await api.get<HoldDetailResponse>(`/holds/${id}`);
  return data;
}

export async function updateHold(
  id: number,
  update: { hold_type?: string; include_in_fit?: boolean },
): Promise<HoldDetailResponse> {
  const { data } = await api.patch<HoldDetailResponse>(`/holds/${id}`, update);
  return data;
}
