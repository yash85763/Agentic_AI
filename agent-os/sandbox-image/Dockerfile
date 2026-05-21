FROM python:3.11-slim

LABEL maintainer="AgentOS"
LABEL description="Hardened sandbox image for safe execution of agent-generated Python code"

# System deps for numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Data science stack pinned for reproducibility
RUN pip install --no-cache-dir \
    pandas==2.2.0 \
    numpy==1.26.3 \
    scipy==1.12.0 \
    statsmodels==0.14.1 \
    scikit-learn==1.4.0 \
    openpyxl==3.1.2 \
    xlrd==2.0.1 \
    pyarrow==14.0.2 \
    duckdb==0.10.0 \
    python-dateutil==2.9.0 \
    holidays==0.46 \
    pytz==2024.1

# Non-root sandbox user (uid 1001)
RUN useradd -m -u 1001 -s /bin/bash sandbox

# Create sandbox directory layout:
#   /sandbox/inputs   — input files copied in by SandboxExecutor (read-only overlay)
#   /sandbox/outputs  — code writes results here (tmpfs at runtime)
# Both directories are owned by sandbox user.
RUN mkdir -p /sandbox/inputs /sandbox/outputs \
    && chown -R sandbox:sandbox /sandbox

USER sandbox
WORKDIR /sandbox

# Default command — overridden at runtime with actual code to execute
CMD ["python", "-c", "print('sandbox ready')"]
