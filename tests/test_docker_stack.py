import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
COMPOSE = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_the_image_is_pinned_and_installs_the_locked_cpu_dependencies_only() -> None:
    assert "FROM python:3.11.18-slim-bookworm" in DOCKERFILE
    assert "ghcr.io/astral-sh/uv:0.13.7" in DOCKERFILE
    assert DOCKERFILE.count("uv sync --frozen --no-dev --no-group gpu") == 2
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE


def test_compose_and_ci_run_the_same_database_image_pinned_by_digest() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pinned = re.compile(r"image: (pgvector/pgvector:0\.8\.7-pg16@sha256:[0-9a-f]{64})$", re.M)
    [compose_image] = pinned.findall(COMPOSE)
    [ci_image] = pinned.findall(workflow)
    assert compose_image == ci_image


def test_secrets_weights_and_test_audio_stay_out_of_the_build_context() -> None:
    for excluded in (".env", ".env.*", ".git", "models", "data/testsets"):
        assert excluded in DOCKERIGNORE
    assert "COPY data/kb" in DOCKERFILE and "COPY data/lexicon" in DOCKERFILE
    assert "COPY . " not in DOCKERFILE


def test_the_app_reaches_the_database_service_and_listens_on_localhost_only() -> None:
    assert "@db:5432/" in COMPOSE
    assert '"127.0.0.1:8000:8000"' in COMPOSE
    assert '"127.0.0.1:5432:5432"' in COMPOSE
    assert "condition: service_healthy" in COMPOSE


def test_models_are_mounted_read_only_and_health_waits_for_readiness() -> None:
    assert ":/root/.cache/huggingface/hub:ro" in COMPOSE
    assert "huggingface}:/root/.cache/huggingface:ro" not in COMPOSE
    assert "./models:/app/models:ro" in COMPOSE
    assert "/v1/ready" in COMPOSE
    assert "GEMINI_API_KEY: ${GEMINI_API_KEY}" in COMPOSE
