import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
      },
      "/context": {
        target: "http://127.0.0.1:8765",
      },
      "/audio": {
        target: "http://127.0.0.1:8765",
      },
      "/viz": {
        target: "http://127.0.0.1:8765",
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/lucide-react")) {
            return "lucide";
          }
        },
      },
    },
  },
});
