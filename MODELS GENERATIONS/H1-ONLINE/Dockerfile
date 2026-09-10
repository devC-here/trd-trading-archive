FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

ENV PORT=5000
ENV HOST=0.0.0.0
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["gunicorn", "gui_server:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4"]
