"""MiewID embedding baseline catalog matcher.

Scores candidates by cosine similarity between ear-crop embeddings from
the pretrained MiewID multi-species re-identification model. Each side
uses its strongest catalog sighting; the two side scores are averaged.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol
from uuid import UUID

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from elephant_id.analysis import PreparedEar
from elephant_id.dataset import PhotoStore
from elephant_id.domain import SightingEarPair
from elephant_id.image import BgrImage, decode_image
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching.protocol import CandidateKey, CandidateScores

_INPUT_SIZE = 440
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EarEmbedder(Protocol):
    """Embed one ear crop as a single feature vector."""

    producer_slug: str

    def embed(self, crop: BgrImage) -> NDArray[np.float32]:
        """Return a one-dimensional L2-normalized float embedding."""
        ...


class MiewIdEmbedder:
    """Embed ear crops with the pretrained MiewID multi-species model.

    The model and its heavy dependencies load lazily on the first
    `embed` call, so constructing the embedder is always cheap.
    """

    def __init__(
        self,
        model_id: str = "conservationxlabs/miewid-msv2",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        """Configure lazy model loading without importing torch."""
        self._model_id = model_id
        self._device = device
        self._revision = revision
        self._model: Any = None
        self.producer_slug = f"{model_id.rsplit('/', 1)[-1]}-cosine-v1"
        if revision is not None:
            self.producer_slug += f"-{revision}"

    def _loaded_model(self) -> Any:
        """Return the eval-mode model, loading it on first use."""
        if self._model is None:
            import torch
            from transformers import AutoModel

            if self._device is None:
                self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Loading MiewID model {self._model_id} on {self._device}")
            model = AutoModel.from_pretrained(
                self._model_id,
                revision=self._revision,
                trust_remote_code=True,
            )
            self._model = model.to(self._device).eval()
        return self._model

    def embed(self, crop: BgrImage) -> NDArray[np.float32]:
        """Return the L2-normalized MiewID embedding for one BGR ear crop.

        Preprocessing resizes to 440x440 RGB, scales to `[0, 1]`, and
        applies ImageNet mean/std normalization.
        """
        import torch

        model = self._loaded_model()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (_INPUT_SIZE, _INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        scaled = resized.astype(np.float32) / 255.0
        normalized = (
            scaled - np.asarray(_IMAGENET_MEAN, dtype=np.float32)
        ) / np.asarray(_IMAGENET_STD, dtype=np.float32)
        batch = (
            torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            output = model(batch)
        vector = output.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)
        return vector / np.linalg.norm(vector)


@dataclass(frozen=True, slots=True)
class _SideMatch:
    """One winning catalog ear and its cosine similarity to the query."""

    catalog_ear: PreparedEar
    score: float


class MiewIdMatcher:
    """Score catalog candidates by ear-embedding cosine similarity.

    Each candidate scores as the mean over the left and right sides of
    the maximum cosine similarity between the query ear embedding and
    that side's catalog evidence embeddings.
    """

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        photo_store: PhotoStore,
        embedder: EarEmbedder,
    ) -> None:
        """Initialize with shared ear preparation and an ear embedder."""
        self._prepare = cache(prepare_ears)
        self._photo_store = photo_store
        self._embedder = embedder
        self._embeddings: dict[tuple[UUID, BoundingBox], NDArray[np.float32]] = {}

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one similarity score per catalog candidate."""
        query_left, query_right = self._prepare(query)
        query_embeddings = (self._embed_ear(query_left), self._embed_ear(query_right))
        scores: dict[CandidateKey, float] = {}
        for candidate_key, evidence in catalog.items():
            scores[candidate_key] = self._match_candidate(
                candidate_key,
                query_embeddings,
                evidence,
            )
        logger.debug(
            f"Scored {len(scores)} candidates with "
            f"{len(self._embeddings)} cached ear embeddings"
        )
        return scores

    def _match_candidate(
        self,
        candidate_key: CandidateKey,
        query_embeddings: tuple[NDArray[np.float32], NDArray[np.float32]],
        evidence: tuple[SightingEarPair, ...],
    ) -> float:
        """Return the mean of the independently winning side similarities.

        Raises:
            RuntimeError: If the candidate has no catalog evidence.
        """
        if not evidence:
            raise RuntimeError(f"{candidate_key} has no catalog evidence")
        prepared = tuple(self._prepare(pair) for pair in evidence)
        left = self._match_side(query_embeddings[0], (ears[0] for ears in prepared))
        right = self._match_side(query_embeddings[1], (ears[1] for ears in prepared))
        return (left.score + right.score) / 2.0

    def _match_side(
        self,
        query_embedding: NDArray[np.float32],
        catalog: Iterable[PreparedEar],
    ) -> _SideMatch:
        """Return the strongest cosine match for one ear side."""

        def side_match_key(match: _SideMatch) -> tuple[float, int, int]:
            return (
                match.score,
                -match.catalog_ear.source_photo.sighting_id.int,
                -match.catalog_ear.source_photo.photo_id.int,
            )

        return max(
            (
                _SideMatch(ear, float(query_embedding @ self._embed_ear(ear)))
                for ear in catalog
            ),
            key=side_match_key,
        )

    def _embed_ear(self, ear: PreparedEar) -> NDArray[np.float32]:
        """Return the memoized embedding for one prepared ear crop."""
        key = (ear.source_photo.photo_id, ear.source_box)
        cached = self._embeddings.get(key)
        if cached is not None:
            return cached
        embedding = self._embedder.embed(self._crop(ear))
        self._embeddings[key] = embedding
        return embedding

    def _crop(self, ear: PreparedEar) -> BgrImage:
        """Return the raster ear crop from the ear's source photo.

        Raises:
            ValueError: If the source box lies outside the decoded image.
        """
        image = decode_image(self._photo_store.read(ear.source_photo))
        box = ear.source_box
        crop = image[box.y1 : box.y2, box.x1 : box.x2]
        if crop.size == 0:
            raise ValueError(
                f"Empty ear crop {box.as_tuple()} for photo {ear.source_photo.photo_id}"
            )
        return crop
