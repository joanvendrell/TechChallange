import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173
  },
  optimizeDeps: {
    include: ["@kitware/vtk.js", "globalthis"],
  },
  build: {
  commonjsOptions: {
    transformMixedEsModules: true, //include: ["@kitware/vtk.js"],
    },
  }
});
