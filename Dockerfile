# AVER Benchmark - Cloud Run + AgentBeats
# Agent Verification & Error Recovery Benchmark
#
# Build:   docker build -t aver-benchmark .
# Run:     docker run -p 8010:8010 aver-benchmark

FROM python:3.11-slim

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
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Cloud Run uses PORT env variable (default 8080)
# AgentBeats controller runs on 8010 and manages agent ports
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8010/agents || exit 1

# Run AgentBeats controller (manages the AVER green agent)
CMD ["agentbeats", "run_ctrl"]
