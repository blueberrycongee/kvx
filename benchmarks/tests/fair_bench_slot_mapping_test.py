import importlib.util
from pathlib import Path

import torch

FAIR_BENCH = Path(__file__).resolve().parents[1] / "fair_bench.py"
spec = importlib.util.spec_from_file_location("fair_bench", FAIR_BENCH)
fair_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fair_bench)
build_kvx_like_slot_mapping = fair_bench.build_kvx_like_slot_mapping


def assert_slot_mapping(seq_count, seq_len, block_size, tokens_per_seq):
    slots = build_kvx_like_slot_mapping(
        seq_count, seq_len, block_size, tokens_per_seq, device="cpu"
    )
    expected = seq_count * tokens_per_seq
    assert slots.numel() == expected, "slot_mapping length mismatch"
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = seq_count * max_blocks_per_seq
    assert slots.max().item() < num_blocks * block_size, "slot out of range"
    assert (slots >= 0).all().item(), "slot mapping contains negative slots"


def assert_slot_mapping_raises(seq_count, seq_len, block_size, tokens_per_seq):
    try:
        build_kvx_like_slot_mapping(
            seq_count, seq_len, block_size, tokens_per_seq, device="cpu"
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid tokens_per_seq")


def main():
    assert_slot_mapping(seq_count=2, seq_len=8, block_size=4, tokens_per_seq=1)
    assert_slot_mapping(seq_count=2, seq_len=8, block_size=4, tokens_per_seq=2)
    assert_slot_mapping(seq_count=4, seq_len=16, block_size=8, tokens_per_seq=4)
    assert_slot_mapping(seq_count=1, seq_len=8, block_size=16, tokens_per_seq=1)
    assert_slot_mapping(seq_count=2, seq_len=8, block_size=4, tokens_per_seq=8)
    assert_slot_mapping_raises(
        seq_count=2, seq_len=8, block_size=4, tokens_per_seq=9
    )
    print("fair_bench_slot_mapping_test passed")


if __name__ == "__main__":
    main()
