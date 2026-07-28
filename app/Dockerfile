# syntax=docker/dockerfile:1
FROM python:3.12-slim

# HF_HOME is where model weights land. Setting it *before* the download step is
# what makes the model part of the image rather than a runtime fetch -- the
# workshop is presented offline, so nothing may download live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/srv

WORKDIR /srv

# build-essential is needed by some wheels; purged after install so the image
# does not carry a compiler around.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt \
 && apt-get purge -y --auto-remove build-essential

# Bake the embedding model into the image. ~90MB, downloaded once at build.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('model cached')"

# Only *after* the bake step: force every later load (runtime, and the
# --network none check below) to use the cache with zero network calls.
# Without this, huggingface_hub still does an online freshness HEAD request
# before falling back to cache -- harmless, but it burns ~2 minutes on
# connection retries with the network gone, which is a bad look mid-talk.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app/ /srv/app/
COPY tests/ /srv/tests/
COPY pytest.ini /srv/pytest.ini
COPY docker-compose.yml /srv/docker-compose.yml

# Vendor Swagger UI for the same reason the model is baked in above: FastAPI's
# stock docs page pulls its JS and CSS from cdn.jsdelivr.net and renders blank
# with no network. Fetch once at build, serve locally at runtime. Downloaded
# here rather than committed, so app/static/vendor/ is gitignored.
RUN python - <<'PY'
import urllib.request
from pathlib import Path

base = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5"
target = Path("/srv/app/static/vendor")
target.mkdir(parents=True, exist_ok=True)
for name in ("swagger-ui-bundle.js", "swagger-ui.css"):
    urllib.request.urlretrieve(f"{base}/{name}", target / name)
    print("vendored", name)
PY

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
