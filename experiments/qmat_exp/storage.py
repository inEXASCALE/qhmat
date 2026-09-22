"""Storage-only controls with the realized partition and ranks held fixed."""
from __future__ import annotations


def same_partition_storage(leaves):
    dense_bits = factor_entries = 0
    for leaf in leaves:
        if int(leaf.get("stored", 1)) == 0:
            raise ValueError("infeasible strict-policy leaf has no stored representation")
        m = int(leaf["row1"]) - int(leaf["row0"])
        n = int(leaf["col1"]) - int(leaf["col0"])
        if int(leaf["low_rank"]) and int(leaf["feasible"]):
            factor_entries += (m + n) * int(leaf["rank"])
        else:
            dense_bits += 64 * m * n
    # Singular values are absorbed into the floating factors: no maps or
    # separate column scales are needed. Dense and fallback blocks stay FP64.
    return {
        "stored_dense_bits": dense_bits,
        "retained_factor_entries": factor_entries,
        "same_partition_fp32_bits": dense_bits + 32 * factor_entries,
        "same_partition_fp64_bits": dense_bits + 64 * factor_entries,
    }
