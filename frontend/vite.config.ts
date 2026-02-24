import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_API_TARGET ?? "http://localhost:8100";
const serverPort = parseInt(process.env.VITE_PORT ?? "5173", 10);

export default defineConfig({
  plugins: [react()],
  server: {
    port: serverPort,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
