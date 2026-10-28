ARG HERMES_IMAGE
FROM ${HERMES_IMAGE}
USER root
WORKDIR /opt/arena
COPY pyproject.toml uv.lock /opt/arena/
RUN uv sync --frozen --no-dev --extra tracking
COPY cases/sc35/harness/requirements.txt /tmp/sc35-requirements.txt
RUN uv pip install --python /opt/hermes/.venv/bin/python -r /tmp/sc35-requirements.txt && uv pip install --python /opt/arena/.venv/bin/python -r /tmp/sc35-requirements.txt
COPY evaluator /opt/arena/evaluator
COPY runner/main.py /opt/arena/runner/main.py
COPY cases/sc71/harness/hermes/*.py cases/sc71/harness/hermes/config.yaml /opt/arena/cases/sc71/harness/hermes/
COPY cases/sc35/harness/*.py /opt/arena/cases/sc35/harness/
COPY cases/sc35/acceptance/*.py /opt/arena/cases/sc35/acceptance/
USER 10001:10001
ENTRYPOINT ["python", "/opt/arena/cases/sc35/harness/hermes.py"]
