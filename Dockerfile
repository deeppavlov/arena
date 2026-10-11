ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.13
ARG DOCKER_IMAGE=docker:27.5.1-cli
FROM ${UV_IMAGE} AS uv
FROM ${DOCKER_IMAGE} AS dockercli
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/arena PATH=/opt/arena/.venv/bin:$PATH
WORKDIR /opt/arena
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && useradd --uid 10001 --create-home arena
COPY evaluator ./evaluator
USER 10001:10001

FROM runtime AS public
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY common ./common
COPY prompts ./prompts
COPY cases ./cases
COPY README.md AGENTS.md RPD.md ./
USER 10001:10001
WORKDIR /workspace
CMD ["bash"]

FROM runtime AS evaluator
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
CMD ["python", "-m", "evaluator"]
