FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS base
LABEL org.opencontainers.image.title="Poindexter SWE Mini-Bench Replication"
LABEL org.opencontainers.image.description="Standalone controller: the reference agent loop in Docker sandboxes, graded by the reference evaluator"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDL_DATA=/data PDL_RESULTS=/results
RUN apt-get update && apt-get install -y --no-install-recommends git patch ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
# The Docker CLI only (static, pinned): the controller drives the host's Docker daemon through
# the mounted socket to start agent sandboxes and evaluator containers. It never runs a daemon.
ARG DOCKER_VERSION=28.5.1
RUN set -eux; arch="$(uname -m)"; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_VERSION}.tgz" -o /tmp/docker.tgz; \
    tar -xzf /tmp/docker.tgz -C /tmp docker/docker; install -m 0755 /tmp/docker/docker /usr/local/bin/docker; \
    rm -rf /tmp/docker /tmp/docker.tgz; docker --version
WORKDIR /opt/poindexter
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock && pip check
COPY vendor ./vendor
COPY presets ./presets
COPY settings-evidence ./settings-evidence
COPY bench.py ./
COPY tests ./tests
RUN mkdir -p /results /data
# No USER: the controller needs the Docker socket, which is root-equivalent on the host by nature.
# Pass --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /var/run/docker.sock)" on Linux to keep
# the results directory owned by you (README.md).
ENTRYPOINT ["python", "/opt/poindexter/bench.py"]
CMD ["plan"]

FROM base AS software-delivery
ENV BENCHMARK_PRESET=software-delivery
FROM base AS cyber-operations
ENV BENCHMARK_PRESET=cyber-operations
