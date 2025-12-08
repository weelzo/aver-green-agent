# AVER Benchmark - Docker Image
# Agent Verification & Error Recovery Benchmark
#
# Build:   docker build -t aver-benchmark .
# Run:     docker run --rm aver-benchmark
# With API key: docker run --rm -e OPENROUTER_API_KEY=sk-... aver-benchmark

# =============================================================================
# Stage 1: Builder - Install dependencies with UV
# =============================================================================
FROM python:3.11-slim as builder

# Install UV for fast dependency management
RUN pip install uv

WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e .

# =============================================================================
# Stage 2: Runtime - Minimal image for running benchmark
# =============================================================================
FROM python:3.11-slim as runtime

LABEL maintainer="AVER Research Team"
LABEL description="AVER Benchmark - Agent Verification & Error Recovery"
LABEL version="0.1.0"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash aver

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY tasks/ ./tasks/
COPY scenarios/ ./scenarios/
COPY pyproject.toml ./

# Create results directory
RUN mkdir -p results && chown -R aver:aver /app

# Switch to non-root user
USER aver

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Default environment variables (can be overridden)
ENV OPENROUTER_API_KEY=""
ENV AVER_MODEL="mock"

# Expose port for potential A2A server mode (future)
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.aver.task_suite import TaskSuite; TaskSuite('tasks').load_all_tasks()" || exit 1

# Default command: Run the demo benchmark
ENTRYPOINT ["python", "-m", "src.aver.cli"]
CMD ["scenarios/aver/scenario.toml"]
