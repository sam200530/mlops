# Deploying the API

The API serves `/health`, `/predict` and `/explain` from the bundled model
artifact. Two images exist because the artifact has to reach the container
somehow, and how it does depends on the host.

| Image | Artifact | Use when |
|---|---|---|
| `Dockerfile` | Mounted at runtime | You control the host and can attach a volume. Retraining does not require an image rebuild. |
| `Dockerfile.deploy` | Baked into the image | The host gives no writable volume (most free tiers). Every retrain means a rebuild. |

## Hugging Face Spaces (recommended for a public demo)

Free, permanent, no card required, and the 22.6 MB artifact can live in the
Space repository.

1. Create a Space at <https://huggingface.co/new-space>, SDK **Docker**, blank template.
2. Clone it and copy the project in:

```bash
git clone https://huggingface.co/spaces/<user>/<space> && cd <space>
cp -r ../mlops/{src,api,configs,requirements.txt} .
cp ../mlops/Dockerfile.deploy ./Dockerfile
mkdir -p models && cp ../mlops/models/model_artifact.pkl models/
```

3. Spaces requires a `README.md` with front matter:

```
---
title: Fraud Detection API
sdk: docker
app_port: 7860
---
```

4. The artifact exceeds Hugging Face's plain-git limit, so track it with LFS:

```bash
git lfs install && git lfs track "*.pkl"
git add .gitattributes . && git commit -m "Deploy fraud detection API" && git push
```

The Space builds and serves at
`https://<user>-<space>.hf.space`. Check `/health` first, then `/docs` for the
interactive API.

**Free Spaces sleep after ~48 hours idle** and wake on the next request, so the
first call after a quiet period is slow. That is fine for a portfolio link and
not fine for anything real.

## Google Cloud Run (closer to a production setup)

Scales to zero, has a genuine free tier, and gives a stable HTTPS URL. Needs a
GCP account with billing enabled even while staying inside the free allowance.

```bash
gcloud run deploy fraud-api \
  --source . --region <region> --allow-unauthenticated --port 7860
```

Cloud Run injects `$PORT`, which `Dockerfile.deploy` already honours. For a real
deployment, keep the artifact out of the image: push it to a bucket and mount it
with Cloud Storage FUSE, which restores the retrain-without-rebuild property.

## You do not need Docker installed

The host builds the image from `Dockerfile.deploy`; nothing is built on your
machine. Hugging Face Spaces builds on push, Cloud Run builds via Cloud Build,
and this project's GitHub Actions workflow has been building the image on every
push already. Docker Engine and the image format are open source and free; only
Docker Desktop has a paid tier, and it is free for personal, educational and
small-company use in any case.

## Optional local check (only if you have Docker)

Skip this entirely if you do not. The hosted build is the same build.

```bash
docker build -f Dockerfile.deploy -t fraud-api .
docker run --rm -p 7860:7860 fraud-api
curl -fsS http://localhost:7860/health
```

`/health` returns `status: "ok"` with the model name, training timestamp and
feature count. It returns `status: "degraded"` with HTTP 200 when no artifact is
present, so the container is reported healthy as a process even before a model
exists -- and `/predict` then returns 503 rather than a confusing error.

## What is deliberately not here

No Kubernetes, autoscaling policy, A/B routing or shadow deployment. The project
is single-node by design and says so in the README's limitations. Adding a
manifest that has never run would be worse than not having one.
