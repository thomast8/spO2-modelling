import { create } from "zustand";
import type { FitPreviewResponse, HoldType } from "../api/types";

interface AppState {
  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Active fit preview (not yet saved)
  fitPreview: FitPreviewResponse | null;
  fitHoldType: HoldType | null;
  fitHoldIds: number[];
  setFitPreview: (preview: FitPreviewResponse, holdType: HoldType, holdIds: number[]) => void;
  clearFitPreview: () => void;

  // Current session context
  activeSessionId: number | null;
  setActiveSessionId: (id: number | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  fitPreview: null,
  fitHoldType: null,
  fitHoldIds: [],
  setFitPreview: (preview, holdType, holdIds) =>
    set({ fitPreview: preview, fitHoldType: holdType, fitHoldIds: holdIds }),
  clearFitPreview: () => set({ fitPreview: null, fitHoldType: null, fitHoldIds: [] }),

  activeSessionId: null,
  setActiveSessionId: (id) => set({ activeSessionId: id }),
}));
