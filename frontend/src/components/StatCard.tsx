import { Box, Paper, Typography } from "@mui/material";
import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  color?: string;
}

export default function StatCard({ label, value, icon, color = "primary.main" }: StatCardProps) {
  return (
    <Paper
      sx={{
        p: 2.5,
        display: "flex",
        alignItems: "center",
        gap: 2,
      }}
    >
      {icon && (
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: `${typeof color === "string" && color.includes(".") ? "" : ""}rgba(79,195,247,0.1)`,
            color,
          }}
        >
          {icon}
        </Box>
      )}
      <Box>
        <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", letterSpacing: 0.5 }}>
          {label}
        </Typography>
        <Typography variant="h5" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
          {value}
        </Typography>
      </Box>
    </Paper>
  );
}
