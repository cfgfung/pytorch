"""Benchmark the CPU FlexAttention SMT sibling-adjacent scheduling.

This compares the default CPU FlexAttention prefill kernel against the variant
emitted when `torch._inductor.config.cpp.flex_attention_smt_pairing` is on. The
pairing schedules SMT sibling threads (logical threads 2p and 2p+1) onto
adjacent q-blocks of the same (batch, head) so the shared K/V stays hot in
L1D/L2 and one sibling's GEMM overlaps the other's softmax on the disjoint
AMX/AVX units. See torch/_inductor/codegen/cpp_flex_attention_template.py.

The two variants generate different C++, so we toggle the config and call
`torch._dynamo.reset()` between them to force a recompile.

For the pairing to do anything, the threads must be bound compactly so that
logical threads 2p and 2p+1 land on the two SMT siblings of one physical core.
Set this in the environment BEFORE launching (it cannot be changed once the
thread pool is up), e.g. on SPR/GNR with GNU OpenMP:

    OMP_NUM_THREADS=$(nproc --all) \
    OMP_PROC_BIND=close OMP_PLACES=cores \
    python benchmarks/transformer/flex_attention_smt_cpu.py

OMP_PLACES=cores (one place per physical core) with close binding and a thread
count of 2x the physical cores packs exactly 2 consecutive threads onto each
core's siblings, giving the 2p/2p+1 pairing the kernel assumes. Do NOT use
OMP_PLACES=threads: that maps threads onto the strided Linux CPU numbering
(sibling of cpu p is p + n_physical), so threads 0 and 1 land on different
cores and the shared-K/V benefit is lost. With Intel OpenMP use the equivalent
compact binding: KMP_AFFINITY=granularity=fine,compact,1,0.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# Changing the mask/score across calls would otherwise recompile repeatedly.
torch._dynamo.config.recompile_limit = 1000


@dataclass
class Config:
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype
    mask: str  # "noop" or "causal"

    @property
    def shape(self):
        return (self.batch, self.heads, self.seq_len, self.head_dim)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def make_inputs(cfg: Config):
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*cfg.shape, dtype=cfg.dtype, device="cpu") for _ in range(3)
    )
    block_mask = None
    if cfg.mask == "causal":
        # block_mask is shape-independent of batch/head here, broadcast over them.
        block_mask = create_block_mask(
            causal_mask, B=None, H=None, Q_LEN=cfg.seq_len, KV_LEN=cfg.seq_len
        )
    return q, k, v, block_mask


def time_fn(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    # Wall-clock is what we care about on CPU; no device sync needed.
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters * 1e3  # ms/iter


def run_variant(cfg: Config, smt_pairing: bool, iters: int, warmup: int):
    q, k, v, block_mask = make_inputs(cfg)

    torch._dynamo.reset()  # force a recompile so the config change takes effect
    with inductor_config.patch({"cpp.flex_attention_smt_pairing": smt_pairing}):
        compiled = torch.compile(flex_attention)

        def run():
            return compiled(q, k, v, block_mask=block_mask)

        out = run()  # triggers compilation
        ms = time_fn(run, iters, warmup)
    return out.float(), ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--mask", choices=["noop", "causal"], default="noop")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    cfg = Config(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        dtype=getattr(torch, args.dtype),
        mask=args.mask,
    )

    num_threads = torch.get_num_threads()
    print(f"torch.get_num_threads() = {num_threads}")
    if num_threads % 2 != 0:
        print(
            "WARNING: odd thread count -> SMT pairing falls back to the plain "
            "split; set an even OMP_NUM_THREADS to exercise pairing."
        )
    print(f"config: shape(B,H,S,D)={cfg.shape} dtype={args.dtype} mask={args.mask}")
    print(f"iters={args.iters} warmup={args.warmup}\n")

    base_out, base_ms = run_variant(cfg, False, args.iters, args.warmup)
    smt_out, smt_ms = run_variant(cfg, True, args.iters, args.warmup)

    # Both variants run identical per-block math in identical order within a
    # block; only the assignment of blocks to threads differs. Outputs should
    # therefore be numerically identical -- a mismatch signals a scheduling bug.
    mismatch = (base_out - smt_out).abs().max().item()
    max_abs = base_out.abs().max().item()
    ok = torch.allclose(base_out, smt_out, rtol=0, atol=0)

    print(f"{'variant':<24}{'ms/iter':>12}{'speedup':>12}")
    print(f"{'baseline':<24}{base_ms:>12.4f}{1.0:>12.3f}")
    print(f"{'smt_pairing':<24}{smt_ms:>12.4f}{base_ms / smt_ms:>12.3f}")
    print(
        f"\ncorrectness: exact_match={ok} max_abs_diff={mismatch:.3e} "
        f"(output max_abs={max_abs:.3e})"
    )
    if not ok:
        print(
            "NOTE: nonzero diff is unexpected for this change; investigate the "
            "block->thread mapping before trusting the timings."
        )


if __name__ == "__main__":
    main()
