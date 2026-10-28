ARG ARENA_BASE_IMAGE
FROM ${ARENA_BASE_IMAGE}
USER root
RUN uv pip install --python /opt/arena/.venv/bin/python lxml==6.1.3 jsonschema==4.26.0
USER 10001:10001
