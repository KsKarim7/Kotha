import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class Embedder:
    def __init__(self, device: str = "cpu") -> None:
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
        # imported here so that code using a fake embedder does not load torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, device=device, local_files_only=True
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)
