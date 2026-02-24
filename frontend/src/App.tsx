import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AboutModelPage from "./pages/AboutModelPage";
import AnalysisPage from "./pages/AnalysisPage";
import DashboardPage from "./pages/DashboardPage";
import ModelsPage from "./pages/ModelsPage";
import UploadPage from "./pages/UploadPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/about-model" element={<AboutModelPage />} />
      </Route>
    </Routes>
  );
}
