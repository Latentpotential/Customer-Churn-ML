# 1. Use the official lightweight Python base image
FROM python:3.14.5-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# Materialize the model at /app/model — this is the path inference.py loads from
# (MODEL_DIR = "/app/model"). It needs the MLflow model dir + feature_columns.txt.
# TODO: point these at a real artifacts path. The model currently lives under
#   mlruns/903599284137618157/models/m-<hash>/artifacts/  (pick the run you want).
# COPY mlruns/<exp>/models/m-<hash>/artifacts/model /app/model
# COPY <path>/feature_columns.txt /app/model/feature_columns.txt

# PYTHONUNBUFFERED=1 ensures logs are shown in real-time (no buffering).
ENV PYTHONUNBUFFERED=1

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]