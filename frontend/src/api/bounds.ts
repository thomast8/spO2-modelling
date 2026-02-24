import api from "./client";
import type { BoundsOverride, BoundsResponse } from "./types";

export async function getBounds(holdType: string): Promise<BoundsResponse> {
  const { data } = await api.get<BoundsResponse>(`/bounds/${holdType}`);
  return data;
}

export async function updateBounds(
  holdType: string,
  bounds: Record<string, BoundsOverride>,
): Promise<BoundsResponse> {
  const { data } = await api.put<BoundsResponse>(`/bounds/${holdType}`, { bounds });
  return data;
}
