from dataclasses import dataclass
from typing import Any


@dataclass
class TrtLlmCacheView:
    """Normalized view of TensorRT-LLM KV cache inputs for KVX adapters."""

    kv_block_offsets: Any
    pool_primary: Any
    pool_secondary: Any
    bytes_per_block: int
    tokens_per_block: int
    num_kv_heads: int
    head_dim: int
    kv_cache_dtype: str


def build_kvx_descriptors(view: TrtLlmCacheView) -> dict:
    """Build KVX descriptors from a TensorRT-LLM cache view."""
    raise NotImplementedError("TODO: implement TensorRT-LLM -> KVX mapping")
