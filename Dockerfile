# AVER Benchmark - Cloud Run + AgentBeats
# Agent Verification & Error Recovery Benchmark
#
# Build:   docker build -t aver-benchmark .
# Run:     docker run -p 8010:8010 aver-benchmark

FROM python:3.13-slim

LABEL maintainer="AVER Research Team"
LABEL description="AVER Benchmark - Agent Verification & Error Recovery"
LABEL version="0.1.0"

WORKDIR /app

# Install dependencies from requirements.txt (Cloud Run standard)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ./
COPY run.sh ./
COPY src/ ./src/
COPY tasks/ ./tasks/
COPY scenarios/ ./scenarios/
COPY pyproject.toml ./

# Make run.sh executable
RUN chmod +x run.sh

# Create results directory
RUN mkdir -p results

# Environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Default AVER server port
EXPOSE 9000

# Health check (default port 9000 per main.py)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Run AVER server (A2A SDK with Starlette)
CMD ["python", "main.py", "run"]
