# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend + static files
FROM python:3.12-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
COPY backend/pyproject.toml backend/uv.lock* ./backend/
WORKDIR /app/backend
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Copy backend code
COPY backend/ ./

# Copy built frontend into backend/static
COPY --from=frontend-build /app/frontend/dist /app/backend/static

# Create data directory for SQLite
RUN mkdir -p /app/data

WORKDIR /app/backend
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
