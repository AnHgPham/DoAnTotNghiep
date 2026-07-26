"""Knowledge-distillation helpers for precomputed teacher embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def normalize_path_key(path: str | Path) -> str:
    """Normalize path keys so Windows and Colab shards can be matched."""
    return Path(path).as_posix()


class TeacherEmbeddingStore:
    """Lookup table for precomputed teacher embeddings.

    Shards are ``.pt`` files containing a dict with ``paths`` and ``embeddings``.
    The store is intentionally memory-resident because teacher vectors are small
    compared with the audio dataset and are hit every training step.
    """

    def __init__(self, root: str | Path, normalize: bool = True):
        self.root = Path(root)
        self.normalize = bool(normalize)
        self._embeddings: dict[str, torch.Tensor] = {}
        self.embedding_dim: int | None = None
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Teacher embedding directory not found: {self.root}")
        shard_paths = sorted(self.root.glob("*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No teacher embedding shards found in {self.root}")

        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            paths = shard.get("paths")
            embeddings = shard.get("embeddings")
            if paths is None or embeddings is None:
                raise ValueError(f"Invalid teacher shard: {shard_path}")
            embeddings = embeddings.float().cpu()
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            self.embedding_dim = int(embeddings.shape[-1])
            if len(paths) != len(embeddings):
                raise RuntimeError("Teacher returned an unexpected embedding count")
            for path, emb in zip(paths, embeddings):
                key = normalize_path_key(path)
                self._embeddings[key] = emb
                try:
                    self._embeddings[Path(path).resolve().as_posix()] = emb
                except OSError:
                    pass

    def __len__(self) -> int:
        return len(self._embeddings)

    def get_many(self, paths: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
        missing = [p for p in paths if normalize_path_key(p) not in self._embeddings]
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(f"Missing teacher embeddings for {len(missing)} paths: {preview}")
        embs = [self._embeddings[normalize_path_key(p)] for p in paths]
        return torch.stack(embs).to(device)


def kd_cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Cosine embedding distillation loss for normalized vectors."""
    if student.shape[-1] != teacher.shape[-1]:
        raise ValueError(
            f"KD dimension mismatch: student={student.shape[-1]}, teacher={teacher.shape[-1]}"
        )
    student = F.normalize(student, p=2, dim=-1)
    teacher = F.normalize(teacher, p=2, dim=-1)
    return 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
