import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#2563eb",
      light: "#60a5fa",
      dark: "#1d4ed8",
    },
    secondary: {
      main: "#d97706",
      light: "#fbbf24",
      dark: "#b45309",
    },
    background: {
      default: "#faf8f5",
      paper: "#ffffff",
    },
    success: {
      main: "#16a34a",
    },
    warning: {
      main: "#d97706",
    },
    error: {
      main: "#dc2626",
    },
    text: {
      primary: "#1c1917",
      secondary: "#78716c",
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica Neue", Arial, sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          boxShadow: "0 1px 3px rgba(28,25,23,0.08)",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 8,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

// Plotly light theme layout defaults
export const plotlyLayout: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: {
    family: "Inter, Roboto, sans-serif",
    color: "#1c1917",
    size: 12,
  },
  xaxis: {
    gridcolor: "rgba(120,113,108,0.12)",
    zerolinecolor: "rgba(120,113,108,0.2)",
  },
  yaxis: {
    gridcolor: "rgba(120,113,108,0.12)",
    zerolinecolor: "rgba(120,113,108,0.2)",
  },
  margin: { l: 60, r: 20, t: 40, b: 50 },
};

// Color palette for chart traces
export const chartColors = {
  spo2Data: "#4fc3f7",
  spo2Fit: "#2563eb",
  hr: "#16a34a",
  threshold: "#dc2626",
  residual: "#d97706",
  confidence: "rgba(37, 99, 235, 0.12)",
};

// SpO2 value-based colorscale: green (high) → yellow → orange → red (low)
export const spo2Colorscale: [number, string][] = [
  [0, "#ef5350"],
  [0.33, "#ff7043"],
  [0.66, "#ffa726"],
  [1, "#66bb6a"],
];
