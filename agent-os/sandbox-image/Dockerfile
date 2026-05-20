FROM python:3.11-slim

LABEL maintainer="AgentOS"
LABEL description="Sandbox image for safe execution of agent-generated Python code"

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

# Non-root user for security
RUN useradd -m -u 1001 -s /bin/bash sandbox \
    && mkdir -p /data /out /home/sandbox \
    && chown -R sandbox:sandbox /data /out /home/sandbox

USER sandbox
WORKDIR /home/sandbox

# Default command — gets overridden at runtime with the actual code to execute
CMD ["python", "-c", "print('sandbox ready')"]
