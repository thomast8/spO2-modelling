import { create } from "zustand";

interface AppState {
  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Current session context
  activeSessionId: number | null;
  setActiveSessionId: (id: number | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  activeSessionId: null,
  setActiveSessionId: (id) => set({ activeSessionId: id }),
}));
