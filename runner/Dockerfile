FROM python:3.12-slim
WORKDIR /opt/runner
COPY runner/main.py /opt/runner/main.py
USER 10001:10001
ENTRYPOINT ["python", "-I", "/opt/runner/main.py"]
