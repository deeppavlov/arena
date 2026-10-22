ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends squid iptables kmod ca-certificates && rm -rf /var/lib/apt/lists/*
COPY squid.conf /etc/squid/squid.conf
COPY firewall.sh /opt/firewall.sh
CMD ["squid", "-N", "-f", "/etc/squid/squid.conf"]
