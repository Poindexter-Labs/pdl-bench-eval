# PDL-Bench replication :: agent sandbox for the executable repository tasks.
#
# The same recipe as the public task image (environments/swe/Dockerfile: the
# base pinned by digest, git + patch, pytest 9.1.1) plus what the reference
# runner's Docker backend expects of an episode container: a non-root `agent`
# user that owns /workspace, `procps` for the stray-process audit, and a
# `sleep infinity` keeper process the runner drives with `docker exec`.
# Grading never happens in this image; the evaluator uses pdl-bench/swe:1.0.0.
#
#   docker build -t pdl-bench/swe-agent:1.0.0 sandbox/swe
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch procps \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "pytest==9.1.1" \
 && useradd -m agent \
 && mkdir -p /workspace/repo \
 && chown -R agent /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace/repo
CMD ["sleep", "infinity"]
