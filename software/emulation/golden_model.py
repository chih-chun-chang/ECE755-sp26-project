"""
Golden Reference / Behavioral Model for Reformer LSH Bucketing+Sorting Accelerator
ECE 755 SP26 - Review 2

This models bit-accurately the full standalone compute block:
  Stage 1: 5x parallel ArgMax units (each scans b 3-bit values -> 3-bit bucket ID)
  Stage 2: Counting-sort Sorting Unit (5 bucket IDs -> sorted (index,bucket) stream)

Design parameters (match RTL):
  N_VEC = 5     # number of projected vectors processed in parallel
  B     = 8     # number of hash buckets (3-bit bucket id)
  W     = 3     # bit-width of each projected component (unsigned)

Matches RTL semantics:
  - ArgMax updates registers ONLY on strictly-greater (>) -> first occurrence wins.
  - After reset, max=0, b_i=0. Until a value > 0 arrives, bucket stays 0.
  - Sorting: bucket_cnt iterates 0..B-1. Each cycle the priority encoder
    picks the lowest-index valid entry whose bucket == bucket_cnt; that entry
    is emitted and its valid is cleared. If none match, bucket_cnt increments.
"""

import random
from typing import List, Tuple

N_VEC = 5
B = 8
W = 3
VMAX = (1 << W) - 1   # 7


# --------------------------- ArgMax sub-block ---------------------------
def argmax_unit(values: List[int]) -> Tuple[int, int]:
    """Bit-accurate single ArgMax unit.
    Returns (max_val, bucket_id). Matches RTL: strictly-greater update.
    """
    assert all(0 <= v <= VMAX for v in values), "3-bit unsigned expected"
    max_val = 0
    b_i = 0
    for idx, v in enumerate(values):
        if v > max_val:
            max_val = v
            b_i = idx
    return max_val, b_i


def argmax_stage(vectors: List[List[int]]) -> List[int]:
    """Run N_VEC parallel ArgMax units. vectors shape: (N_VEC, B)."""
    assert len(vectors) == N_VEC
    return [argmax_unit(v)[1] for v in vectors]


# --------------------------- Sorting sub-block ---------------------------
def sorting_unit(bucket_ids: List[int]) -> List[Tuple[int, int]]:
    """Counting-sort style. Emits (vec_index, bucket) pairs in ascending
    bucket order. Ties broken by ascending vec_index (priority encoder)."""
    assert len(bucket_ids) == N_VEC
    valid = [True] * N_VEC
    out = []
    for b in range(B):
        # keep emitting as long as something matches current bucket count
        while True:
            sel = None
            for i in range(N_VEC):
                if valid[i] and bucket_ids[i] == b:
                    sel = i
                    break
            if sel is None:
                break
            out.append((sel, b))
            valid[sel] = False
        if not any(valid):
            break
    return out


# --------------------------- Top-level block ---------------------------
def top_block(vectors: List[List[int]]) -> List[Tuple[int, int]]:
    return sorting_unit(argmax_stage(vectors))


# --------------------------- Cycle-count model --------------------------
def cycle_counts(vectors: List[List[int]]):
    """Approximate cycle counts.
    ArgMax phase: B data-clock cycles (one per component).
    Sorting phase: (#emitted entries) + (#bucket increments with no match)
                   <= N_VEC + B system-clock cycles.
    """
    bids = argmax_stage(vectors)
    emit = 0
    incs = 0
    valid = [True] * N_VEC
    for b in range(B):
        matched_any = False
        while True:
            sel = None
            for i in range(N_VEC):
                if valid[i] and bids[i] == b:
                    sel = i
                    break
            if sel is None:
                break
            matched_any = True
            valid[sel] = False
            emit += 1
        if not matched_any:
            incs += 1
        if not any(valid):
            break
    return {"argmax_cycles": B, "sort_cycles": emit + incs,
            "emitted": emit, "empty_bucket_steps": incs}


# --------------------------- Unit tests (deterministic) ----------------
def _run_unit_tests():
    print("==== ArgMax unit tests ====")
    cases = [
        ([1,2,3,4,5,6,7,0], (7,6), "ascending then zero"),
        ([7,6,5,4,3,2,1,0], (7,0), "descending"),
        ([2,5,1,3,0,0,0,0], (5,1), "max in middle"),
        ([4,4,4,4,4,4,4,4], (4,0), "all-equal -> first wins"),
        ([0,0,0,0,0,0,0,0], (0,0), "all-zero -> default"),
        ([0,0,0,0,0,0,0,6], (6,7), "max at last"),
    ]
    ok = True
    for vec, exp, label in cases:
        got = argmax_unit(vec)
        status = "PASS" if got == exp else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {label:35s} got={got} exp={exp}")

    print("\n==== Sorting unit tests ====")
    stests = [
        ([0,0,0,0,0],
         [(0,0),(1,0),(2,0),(3,0),(4,0)], "all same bucket"),
        ([7,0,3,1,2],
         [(1,0),(3,1),(4,2),(2,3),(0,7)], "distinct buckets"),
        ([2,2,5,5,2],
         [(0,2),(1,2),(4,2),(2,5),(3,5)], "two groups"),
        ([0,1,2,3,4],
         [(0,0),(1,1),(2,2),(3,3),(4,4)], "already sorted"),
    ]
    for bids, exp, label in stests:
        got = sorting_unit(bids)
        status = "PASS" if got == exp else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {label:25s} got={got}")

    print("\n==== Top-level random test ====")
    random.seed(0xC001)
    fails = 0
    for t in range(200):
        vectors = [[random.randint(0, VMAX) for _ in range(B)] for _ in range(N_VEC)]
        bids = argmax_stage(vectors)
        sorted_out = top_block(vectors)
        # invariant 1: sorted buckets non-decreasing
        buckets_only = [b for _, b in sorted_out]
        if buckets_only != sorted(buckets_only):
            fails += 1
        # invariant 2: multiset of (idx, bucket) matches bids assignment
        expect_pairs = sorted([(i, bids[i]) for i in range(N_VEC)],
                              key=lambda p: (p[1], p[0]))
        if sorted_out != expect_pairs:
            fails += 1
    print(f"  random trials: 200, failures: {fails}")
    ok &= (fails == 0)

    cc = cycle_counts([[random.randint(0, VMAX) for _ in range(B)] for _ in range(N_VEC)])
    print(f"\nCycle-count sample: {cc}")
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return ok


if __name__ == "__main__":
    _run_unit_tests()
