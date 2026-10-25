ARG PYTHON_IMAGE=python:3.14-slim
FROM ${PYTHON_IMAGE}
ARG HERMES_REVISION=86fa9e6ca1de7ee71132d2be0103f3e37d1fa872
ARG UV_VERSION=0.8.13
RUN apt-get update && apt-get install -y --no-install-recommends git curl ripgrep ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==${UV_VERSION}
COPY cases/sc71/harness/hermes/source-cache/ /opt/hermes-source-cache/
RUN mkdir /opt/hermes && if [ -f /opt/hermes-source-cache/source.tar ]; then \
      test "${HERMES_REVISION}" = "86fa9e6ca1de7ee71132d2be0103f3e37d1fa872" && \
      tar -xf /opt/hermes-source-cache/source.tar -C /opt/hermes; \
    else \
      git -C /opt/hermes init && \
      git -C /opt/hermes remote add origin https://github.com/NousResearch/hermes-agent.git && \
      git -C /opt/hermes fetch --depth 1 origin ${HERMES_REVISION} && \
      git -C /opt/hermes checkout --detach FETCH_HEAD && \
      test "$(git -C /opt/hermes rev-parse HEAD)" = "${HERMES_REVISION}"; \
    fi && rm -rf /opt/hermes-source-cache && echo "${HERMES_REVISION}" > /opt/hermes/.arena-revision
WORKDIR /opt/hermes
RUN uv sync --frozen --no-dev --extra ddgs
RUN ln -s /assets/kb /kb
WORKDIR /opt/arena
COPY pyproject.toml uv.lock /opt/arena/
RUN uv sync --frozen --no-dev --extra tracking
COPY evaluator /opt/arena/evaluator
COPY runner/main.py /opt/arena/runner/main.py
COPY cases/sc71/harness/hermes/*.py cases/sc71/harness/hermes/config.yaml /opt/arena/cases/sc71/harness/hermes/
ENV PATH="/opt/hermes/.venv/bin:${PATH}"
ENV PYTHONPATH="/opt/arena:/opt/hermes"
ENV HOME="/tmp/arena-home"
ENV HERMES_HOME="/tmp/arena-hermes-home"
ENV PYTHONDONTWRITEBYTECODE="1"
USER 10001:10001
ENTRYPOINT ["python", "/opt/arena/cases/sc71/harness/hermes/main.py"]
