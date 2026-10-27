# PDL-Bench replication :: agent sandbox for the question-answering tasks.
#
# The public SWEQnA image recipe (environments/sweqna/Dockerfile) plus the
# `agent` user, procps and the keeper process the reference runner expects.
# PyYAML is NOT installed here: the agent writes prose; the property gate
# that imports yaml runs in the evaluator image, never in the agent sandbox.
#
#   docker build -t pdl-bench/sweqna-agent:1.0.0 sandbox/sweqna
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git procps \
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
