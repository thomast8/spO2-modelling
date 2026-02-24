import { Routes, Route } from "react-router-dom";
import { Box, Typography } from "@mui/material";

function Placeholder({ title }: { title: string }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
      }}
    >
      <Typography variant="h4" color="text.secondary">
        {title} — Coming Soon
      </Typography>
    </Box>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Placeholder title="SpO2 Modelling Dashboard" />} />
      <Route path="/upload" element={<Placeholder title="Upload Session" />} />
      <Route path="/fit" element={<Placeholder title="Fit Model" />} />
      <Route path="/models" element={<Placeholder title="Model Versions" />} />
      <Route path="/analysis" element={<Placeholder title="Analysis" />} />
    </Routes>
  );
}
