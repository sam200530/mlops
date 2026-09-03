#!/bin/sh
# Fetch the model artifact on startup, then hand off to uvicorn.
#
# The artifact is not in the repository (.gitignore excludes model weights) and
# managed hosts give no writable volume to mount one into, so it is downloaded
# at container start from $MODEL_URL. Doing it here rather than at build time
# means a new model is a restart, not a rebuild -- the same property the
# volume-mounted Dockerfile has.
set -e

ARTIFACT=/app/models/model_artifact.pkl

if [ -f "$ARTIFACT" ]; then
    echo "entrypoint: artifact already present, skipping download"
elif [ -n "$MODEL_URL" ]; then
    echo "entrypoint: downloading artifact from \$MODEL_URL"
    mkdir -p /app/models
    curl -fsSL "$MODEL_URL" -o "$ARTIFACT"
    SIZE=$(wc -c < "$ARTIFACT")
    if [ "$SIZE" -lt 1000000 ]; then
        echo "entrypoint: ERROR downloaded file is only ${SIZE} bytes."
        echo "entrypoint: MODEL_URL probably points at an HTML page, not the asset."
        exit 1
    fi
    echo "entrypoint: artifact ready (${SIZE} bytes)"
else
    # Not fatal: /health reports "degraded" and the prediction routes return
    # 503, which is more useful than refusing to start.
    echo "entrypoint: WARNING no MODEL_URL set and no artifact on disk."
    echo "entrypoint: starting anyway -- /health will report degraded."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
