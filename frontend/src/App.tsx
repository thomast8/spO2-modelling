import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AnalysisPage from "./pages/AnalysisPage";
import DashboardPage from "./pages/DashboardPage";
import FitPage from "./pages/FitPage";
import ModelsPage from "./pages/ModelsPage";
import UploadPage from "./pages/UploadPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/fit" element={<FitPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
      </Route>
    </Routes>
  );
}
