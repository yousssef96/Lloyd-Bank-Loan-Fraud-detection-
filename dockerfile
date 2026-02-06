# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/model /app/model
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/preprocessing.pkl /app/model/preprocessing.pkl


# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]