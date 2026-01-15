from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TgiCacheView:
    """Normalized view of TGI KV cache inputs for KVX adapters."""

    k_cache: Any
    v_cache: Any
    block_tables_tensor: Any
    block_tables_ragged: Optional[Any]
    slots: Any
    block_size: int
    num_kv_heads: int
    head_dim: int
    kv_cache_dtype: str
    layout: str


def build_kvx_descriptors(view: TgiCacheView) -> dict:
    """Build KVX descriptors from a TGI cache view."""
    raise NotImplementedError("TODO: implement TGI -> KVX descriptor mapping")
