import { createTheme } from "@mui/material/styles";

export const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#4fc3f7",
      light: "#8bf6ff",
      dark: "#0093c4",
    },
    secondary: {
      main: "#ff7043",
      light: "#ffa270",
      dark: "#c63f17",
    },
    background: {
      default: "#0a0e17",
      paper: "#111827",
    },
    success: {
      main: "#66bb6a",
    },
    warning: {
      main: "#ffa726",
    },
    error: {
      main: "#ef5350",
    },
    text: {
      primary: "#e2e8f0",
      secondary: "#94a3b8",
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
          border: "1px solid rgba(255, 255, 255, 0.06)",
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

// Plotly dark theme layout defaults
export const plotlyDarkLayout: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "rgba(17, 24, 39, 0.8)",
  font: {
    family: "Inter, Roboto, sans-serif",
    color: "#e2e8f0",
    size: 12,
  },
  xaxis: {
    gridcolor: "rgba(148, 163, 184, 0.1)",
    zerolinecolor: "rgba(148, 163, 184, 0.2)",
  },
  yaxis: {
    gridcolor: "rgba(148, 163, 184, 0.1)",
    zerolinecolor: "rgba(148, 163, 184, 0.2)",
  },
  margin: { l: 60, r: 20, t: 40, b: 50 },
};

// Color palette for chart traces
export const chartColors = {
  spo2Data: "#4fc3f7",
  spo2Fit: "#ff7043",
  hr: "#66bb6a",
  threshold: "#ef5350",
  residual: "#ffa726",
  confidence: "rgba(79, 195, 247, 0.15)",
};
