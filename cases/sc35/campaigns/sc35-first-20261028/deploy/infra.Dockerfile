ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends squid iptables kmod ca-certificates && rm -rf /var/lib/apt/lists/*
COPY cases/sc71/deploy/gpu6/squid.conf /etc/squid/squid.conf
COPY cases/sc35/campaigns/sc35-first-20261028/deploy/firewall.sh /opt/firewall.sh
CMD ["squid", "-N", "-f", "/etc/squid/squid.conf"]
