"""Test bootstrap: project root on sys.path, stubs for heavy optional imports."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_STUBBED: set[str] = set()


def _module_available(name: str) -> bool:
    loaded = sys.modules.get(name)
    if loaded is not None and not getattr(loaded, "_ARXIVLENS_STUB", False):
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _new_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<arxivlens stub: {name}>"
    module.__path__ = []
    module._ARXIVLENS_STUB = True
    sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)
    return module


class _ChromaSettings:
    def __init__(self, *args, **kwargs):
        pass


class _NotFoundError(Exception):
    """Stand-in for chromadb.errors.NotFoundError when chromadb is not installed."""


class _UnavailableSentenceTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("sentence_transformers is not installed")


def _stub_heavy_imports() -> None:
    """Keep CI off chromadb, sentence-transformers, and torch.

    Those packages are imported at module level. The pure helpers under test
    do not need them, and installing them pulls in large native wheels.
    """
    if not _module_available("chromadb"):
        chromadb = _new_stub("chromadb")
        config = _new_stub("chromadb.config")
        config.Settings = _ChromaSettings
        chromadb.config = config
        errors = _new_stub("chromadb.errors")
        errors.NotFoundError = _NotFoundError
        chromadb.errors = errors
        _STUBBED.add("chromadb")

    if not _module_available("sentence_transformers"):
        sentence_transformers = _new_stub("sentence_transformers")
        sentence_transformers.SentenceTransformer = _UnavailableSentenceTransformer
        sentence_transformers.CrossEncoder = _UnavailableSentenceTransformer
        _STUBBED.add("sentence_transformers")

    if not _module_available("torch"):
        _new_stub("torch")
        _STUBBED.add("torch")


_stub_heavy_imports()


def pytest_collection_modifyitems(config, items):
    if "sentence_transformers" not in _STUBBED:
        return
    skip = pytest.mark.skip(reason="sentence_transformers is not installed")
    for item in items:
        if "EmbeddingModelWindowTests" in item.nodeid:
            item.add_marker(skip)
