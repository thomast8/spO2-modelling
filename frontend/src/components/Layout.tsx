import {
  Assessment as AnalysisIcon,
  CloudUpload as UploadIcon,
  Dashboard as DashboardIcon,
  Menu as MenuIcon,
  ModelTraining as ModelIcon,
} from "@mui/icons-material";
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { useAppStore } from "../store/appStore";

const DRAWER_WIDTH = 240;

const navItems = [
  { label: "Dashboard", path: "/", icon: <DashboardIcon /> },
  { label: "Upload & Fit", path: "/upload", icon: <UploadIcon /> },
  { label: "Models", path: "/models", icon: <ModelIcon /> },
  { label: "Analysis", path: "/analysis", icon: <AnalysisIcon /> },
];

export default function Layout() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const { sidebarOpen, toggleSidebar } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();

  const drawerContent = (
    <Box sx={{ mt: 1 }}>
      <Box sx={{ px: 2.5, py: 2 }}>
        <Typography
          variant="h6"
          sx={{
            background: "linear-gradient(135deg, #2563eb 0%, #d97706 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontWeight: 800,
            letterSpacing: "-0.03em",
            fontSize: "1.25rem",
          }}
        >
          SpO₂ Modelling
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Hill Equation ODC Fit
        </Typography>
      </Box>
      <List sx={{ px: 1 }}>
        {navItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <ListItemButton
              key={item.path}
              onClick={() => {
                navigate(item.path);
                if (isMobile) toggleSidebar();
              }}
              sx={{
                borderRadius: 2,
                mb: 0.5,
                bgcolor: active ? "rgba(37, 99, 235, 0.08)" : "transparent",
                color: active ? "primary.main" : "text.secondary",
                "&:hover": {
                  bgcolor: active ? "rgba(37, 99, 235, 0.12)" : "rgba(28,25,23,0.04)",
                },
              }}
            >
              <ListItemIcon sx={{ color: "inherit", minWidth: 40 }}>{item.icon}</ListItemIcon>
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontWeight: active ? 700 : 500,
                  fontSize: "0.9rem",
                }}
              />
            </ListItemButton>
          );
        })}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      {/* Mobile AppBar */}
      {isMobile && (
        <AppBar
          position="fixed"
          elevation={0}
          sx={{
            bgcolor: "background.paper",
            borderBottom: "1px solid #e7e5e4",
          }}
        >
          <Toolbar>
            <IconButton edge="start" color="inherit" onClick={toggleSidebar} sx={{ mr: 1 }}>
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap sx={{ fontWeight: 700 }}>
              SpO₂ Modelling
            </Typography>
          </Toolbar>
        </AppBar>
      )}

      {/* Sidebar Drawer */}
      <Drawer
        variant={isMobile ? "temporary" : "permanent"}
        open={isMobile ? sidebarOpen : true}
        onClose={toggleSidebar}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: DRAWER_WIDTH,
            boxSizing: "border-box",
            bgcolor: "background.paper",
            borderRight: "1px solid #e7e5e4",
          },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: isMobile ? 8 : 0,
          minWidth: 0,
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}
