ARG HERMES_IMAGE
FROM ${HERMES_IMAGE}
ENTRYPOINT ["python", "/opt/arena/cases/sc35/harness/content.py"]
