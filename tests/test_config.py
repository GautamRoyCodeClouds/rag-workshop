"""Config defaults are the deck's numbers. If one of these fails, either the
deck changed or someone drifted from it -- check the slide first.
"""
from pathlib import Path

import chromadb
import pytest
import yaml

from app.config import Settings


def test_defaults_match_the_deck():
    s = Settings.from_env({})
    # "Sensible defaults for version one" slide (Level 6)
    assert s.default_chunk_size == 700
    assert s.default_chunk_overlap == 100
    assert s.default_strategy == "recursive"
    # Level 2 model table, self-host row
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embed_dims == 384


def test_env_overrides_are_typed():
    s = Settings.from_env({"DEFAULT_CHUNK_SIZE": "1500", "CHROMA_PORT": "9000"})
    assert s.default_chunk_size == 1500
    assert s.chroma_port == 9000
    assert isinstance(s.chroma_port, int)


def test_retrieval_defaults():
    s = Settings.from_env({})
    assert s.retrieval_top_k == 5
    assert s.retrieval_min_score == 0.25
    assert s.retrieval_mmr_lambda == 0.5
    assert s.retrieval_pool_multiplier == 4
    assert s.ollama_model == "deepseek-r1:1.5b"
    assert s.ollama_base_url == ""


def test_retrieval_env_overrides_are_typed():
    s = Settings.from_env({
        "RETRIEVAL_TOP_K": "8",
        "RETRIEVAL_MIN_SCORE": "0.4",
        "RETRIEVAL_MMR_LAMBDA": "0.7",
        "RETRIEVAL_POOL_MULTIPLIER": "6",
        "OLLAMA_MODEL": "llama3.2:1b",
        "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    })
    assert s.retrieval_top_k == 8
    assert isinstance(s.retrieval_top_k, int)
    assert s.retrieval_min_score == pytest.approx(0.4)
    assert isinstance(s.retrieval_min_score, float)
    assert s.retrieval_mmr_lambda == pytest.approx(0.7)
    assert s.retrieval_pool_multiplier == 6
    assert s.ollama_model == "llama3.2:1b"
    assert s.ollama_base_url == "http://host.docker.internal:11434"


def test_decimal_empty_string_falls_back_to_default():
    # Same contract as number()'s empty-string handling, checked for the new
    # float helper specifically -- an empty env var must not become 0.0,
    # which would make min_score's threshold accept everything.
    s = Settings.from_env({"RETRIEVAL_MIN_SCORE": ""})
    assert s.retrieval_min_score == 0.25


def test_decimal_malformed_value_raises():
    # Mirrors number()'s int(value) behaviour: a genuinely bad value must
    # raise, not silently coerce to 0.0 or the default.
    with pytest.raises(ValueError):
        Settings.from_env({"RETRIEVAL_MIN_SCORE": "not-a-number"})


def test_decimal_malformed_mmr_lambda_raises():
    with pytest.raises(ValueError):
        Settings.from_env({"RETRIEVAL_MMR_LAMBDA": "abc"})


def test_empty_env_value_falls_back_to_default():
    assert Settings.from_env({"DEFAULT_CHUNK_SIZE": ""}).default_chunk_size == 700


def test_local_pdf_is_none_when_unset():
    assert Settings.from_env({}).local_pdf is None


def test_local_pdf_is_none_when_path_does_not_exist():
    assert Settings.from_env({"LOCAL_PDF_PATH": "/nope/absent.pdf"}).local_pdf is None


def test_local_pdf_resolves_when_file_exists(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    assert Settings.from_env({"LOCAL_PDF_PATH": str(p)}).local_pdf == Path(p)


def test_max_upload_bytes_computed_from_mb():
    assert Settings.from_env({"MAX_UPLOAD_MB": "2"}).max_upload_bytes == 2 * 1024 * 1024


def test_compose_chroma_server_matches_the_installed_client_version():
    compose_path = Path(__file__).parents[1] / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert compose["services"]["chromadb"]["image"] == (
        f"chromadb/chroma:{chromadb.__version__}"
    )
