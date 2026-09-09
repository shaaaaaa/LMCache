# SPDX-License-Identifier: Apache-2.0
"""Prepared source state for layerwise sparse cache retrieval."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj


_PREPARED_SPARSE_SOURCE_IDS = count(1)


@dataclass(frozen=True, slots=True)
class PreparedSparseSourceLayer:
    """Stable CPU source and pointer table for one sparse cache layer."""

    tensors: tuple[torch.Tensor, ...]
    chunk_ptrs_npu: torch.Tensor
    memory_objs: tuple[MemoryObj, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedSparseSource:
    """Request-owned sparse source resolved once after cache bootstrap."""

    layers: tuple[PreparedSparseSourceLayer, ...]
    total_tokens: int
    chunk_token_counts: tuple[int, ...] = field(default_factory=tuple)
    pointer_device: Optional[torch.device] = None
    binding_id: int = field(
        default_factory=lambda: next(_PREPARED_SPARSE_SOURCE_IDS)
    )


@dataclass(slots=True)
class PreparedSparseGraphStep:
    """Pinned, request-aligned sparse sources for one device-graph replay."""

    request_ids: tuple[str, ...]
    layer_names: tuple[str, ...]
    sources: tuple[PreparedSparseSource, ...]
    request_capacity: int
    _owners: tuple[MemoryObj, ...] = field(default_factory=tuple, repr=False)
    _sources_by_request: dict[str, PreparedSparseSource] = field(
        default_factory=dict,
        repr=False,
    )
    _released: bool = field(default=False, init=False, repr=False)

    @classmethod
    def acquire(
        cls,
        request_ids: Sequence[str],
        layer_names: Sequence[str],
        sources: Sequence[PreparedSparseSource],
        request_capacity: int,
    ) -> "PreparedSparseGraphStep":
        """Pin source owners until graph replay completion is fenced."""
        request_tuple = tuple(request_ids)
        source_tuple = tuple(sources)
        layer_tuple = tuple(layer_names)
        if not request_tuple or len(request_tuple) != len(source_tuple):
            raise ValueError("Graph sources must align with non-empty request IDs")
        if len(request_tuple) > request_capacity:
            raise ValueError("Graph request count exceeds its fixed capacity")
        if not layer_tuple:
            raise ValueError("Graph source layer names must not be empty")
        if any(len(source.layers) < len(layer_tuple) for source in source_tuple):
            raise ValueError("Graph sources must cover every target layer")

        owners: list[MemoryObj] = []
        seen_owner_ids: set[int] = set()
        try:
            for source in source_tuple:
                for layer in source.layers:
                    for owner in layer.memory_objs:
                        owner_id = id(owner)
                        if owner_id in seen_owner_ids:
                            continue
                        if not owner.is_valid():
                            raise RuntimeError(
                                "Sparse graph source owner is no longer valid"
                            )
                        owner.ref_count_up()
                        seen_owner_ids.add(owner_id)
                        owners.append(owner)
        except BaseException:
            for owner in reversed(owners):
                if owner.is_valid():
                    owner.ref_count_down()
            raise
        return cls(
            request_ids=request_tuple,
            layer_names=layer_tuple,
            sources=source_tuple,
            request_capacity=int(request_capacity),
            _owners=tuple(owners),
            _sources_by_request=dict(
                zip(request_tuple, source_tuple, strict=True)
            ),
        )

    def release(self) -> None:
        """Drop graph-owned references after the replay fence completes."""
        if self._released:
            return
        self._released = True
        for owner in reversed(self._owners):
            if owner.is_valid():
                owner.ref_count_down()

    def matches(
        self,
        request_ids: Sequence[str],
        layer_names: Sequence[str],
        sources: Sequence[PreparedSparseSource],
        request_capacity: int,
    ) -> bool:
        """Return whether this pinned binding can be reused without CPU work."""
        return bool(
            not self._released
            and self.request_ids == tuple(request_ids)
            and self.layer_names == tuple(layer_names)
            and self.request_capacity == int(request_capacity)
            and tuple(source.binding_id for source in self.sources)
            == tuple(source.binding_id for source in sources)
        )

    def source_for_request(
        self, request_id: str
    ) -> Optional[PreparedSparseSource]:
        """Look up one lane without rebuilding a per-step request map."""
        return self._sources_by_request.get(request_id)


def build_prepared_sparse_source(
    cached_tensors: Sequence[Sequence[torch.Tensor]],
    cached_chunk_ptrs_npu: Sequence[Optional[torch.Tensor]],
    *,
    num_layers: int,
    total_tokens: int,
    chunk_token_counts: Optional[Sequence[int]] = None,
    expected_pointer_device: Optional[torch.device] = None,
    cached_memory_objs: Optional[Sequence[Sequence[MemoryObj]]] = None,
) -> Optional[PreparedSparseSource]:
    """Seal a complete layer cache into immutable hot-path source metadata.

    Args:
        cached_tensors: CPU chunk tensors in layer-major order.
        cached_chunk_ptrs_npu: NPU pointer tables in layer-major order.
        num_layers: Exact layer count required for a complete binding.
        total_tokens: Number of valid source tokens represented by the cache.
        chunk_token_counts: Request-owned token coverage for each CPU chunk.
        expected_pointer_device: Accelerator device that owns pointer tables.
        cached_memory_objs: Optional pointer-first CPU chunk owners. Complete
            owner layers can replace ``cached_tensors`` without constructing
            per-chunk typed views.

    Returns:
        A prepared source, or ``None`` while bootstrap data is incomplete.

    Raises:
        TypeError: A completed cache contains an object of the wrong type.
        ValueError: Pointer metadata is malformed or has partial coverage.

    Incomplete caches are expected while the bootstrap generator is running and
    return ``None``. Once a layer has a pointer tensor, malformed pointer
    coverage is an invariant violation and is reported immediately.
    """
    if num_layers <= 0 or total_tokens <= 0:
        return None
    tensors_complete = len(cached_tensors) == num_layers
    owners_complete = (
        cached_memory_objs is not None and len(cached_memory_objs) == num_layers
    )
    if not tensors_complete and not owners_complete:
        return None
    if len(cached_chunk_ptrs_npu) != num_layers:
        return None

    normalized_chunk_counts: tuple[int, ...] = ()
    if chunk_token_counts is not None:
        normalized_chunk_counts = tuple(int(count) for count in chunk_token_counts)
        if any(count <= 0 for count in normalized_chunk_counts):
            raise ValueError("Prepared sparse chunk token counts must be positive.")
        covered_tokens = sum(normalized_chunk_counts)
        if covered_tokens < total_tokens:
            return None

    layers: list[PreparedSparseSourceLayer] = []
    pointer_device: Optional[torch.device] = None
    for layer_id in range(num_layers):
        layer_tensors = cached_tensors[layer_id] if tensors_complete else ()
        if isinstance(layer_tensors, torch.Tensor):
            raise TypeError(
                "Prepared sparse source layers must contain tensor sequences: "
                f"layer_id={layer_id}"
            )
        tensors = tuple(layer_tensors)
        memory_objs = (
            tuple(cached_memory_objs[layer_id])
            if cached_memory_objs is not None and owners_complete
            else ()
        )
        chunk_counts = {len(chunks) for chunks in (tensors, memory_objs) if chunks}
        if not chunk_counts:
            return None
        if len(chunk_counts) != 1:
            raise ValueError(
                "Prepared sparse tensor and MemoryObj caches disagree: "
                f"layer_id={layer_id}, tensors={len(tensors)}, "
                f"memory_objs={len(memory_objs)}"
            )
        chunk_count = chunk_counts.pop()
        chunk_ptrs_npu = cached_chunk_ptrs_npu[layer_id]
        if chunk_ptrs_npu is None:
            return None
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(
                "Prepared sparse source contains a non-tensor entry: "
                f"layer_id={layer_id}"
            )
        if not isinstance(chunk_ptrs_npu, torch.Tensor):
            raise TypeError(
                "Prepared sparse pointer cache must contain tensors: "
                f"layer_id={layer_id}, type={type(chunk_ptrs_npu).__name__}"
            )
        if chunk_ptrs_npu.ndim != 1 or chunk_ptrs_npu.dtype != torch.int64:
            raise ValueError(
                "Prepared sparse pointer cache must be a 1D int64 tensor: "
                f"layer_id={layer_id}, shape={tuple(chunk_ptrs_npu.shape)}, "
                f"dtype={chunk_ptrs_npu.dtype}"
            )
        if not chunk_ptrs_npu.is_contiguous():
            raise ValueError(
                "Prepared sparse pointer cache must be contiguous: "
                f"layer_id={layer_id}, stride={chunk_ptrs_npu.stride()}"
            )
        if int(chunk_ptrs_npu.numel()) != chunk_count:
            raise ValueError(
                "Prepared sparse pointer coverage does not match CPU chunks: "
                f"layer_id={layer_id}, pointers={chunk_ptrs_npu.numel()}, "
                f"chunks={chunk_count}"
            )
        if normalized_chunk_counts and len(normalized_chunk_counts) != chunk_count:
            raise ValueError(
                "Prepared sparse chunk coverage does not match CPU chunks: "
                f"layer_id={layer_id}, coverage={len(normalized_chunk_counts)}, "
                f"chunks={chunk_count}"
            )
        if pointer_device is None:
            pointer_device = chunk_ptrs_npu.device
        elif chunk_ptrs_npu.device != pointer_device:
            raise ValueError(
                "Prepared sparse pointer tables must share one device: "
                f"layer_id={layer_id}, device={chunk_ptrs_npu.device}, "
                f"expected={pointer_device}"
            )
        if (
            expected_pointer_device is not None
            and (
                chunk_ptrs_npu.device.type != expected_pointer_device.type
                or (
                    expected_pointer_device.index is not None
                    and chunk_ptrs_npu.device.index != expected_pointer_device.index
                )
            )
        ):
            raise ValueError(
                "Prepared sparse pointer table is on the wrong device: "
                f"layer_id={layer_id}, device={chunk_ptrs_npu.device}, "
                f"expected={expected_pointer_device}"
            )

        layers.append(
            PreparedSparseSourceLayer(
                tensors=tensors,
                chunk_ptrs_npu=chunk_ptrs_npu,
                memory_objs=memory_objs,
            )
        )

    layer_tuple = tuple(layers)
    return PreparedSparseSource(
        layers=layer_tuple,
        total_tokens=int(total_tokens),
        chunk_token_counts=normalized_chunk_counts,
        pointer_device=pointer_device,
    )
