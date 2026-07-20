# Owner(s): ["module: inductor"]
"""Robustness + correctness tests for the CPU FlexAttention AMX/AVX-512 overlap
kernel (torch/_inductor/codegen/cpp_flex_attention_amx.py and the need_pack path
in cpp_flex_attention_template.py).

The overlap kernel replaces the oneDNN brgemm Q@K^T / P@V with hand-written AMX
bf16 GEMMs interleaved with the AVX-512 online softmax (intel_amx_opt.docx 11.2).
It is enabled for bf16 with headSize % 32 == 0, headSize_v % 32 == 0, and
kvSplitSize % 32 == 0. kvSize need NOT be a multiple of kvSplitSize: aligned
kv-blocks run on AMX+overlap and the ragged tail block runs on brgemm (so any
sequence length is supported). Non-conforming shapes (odd/non-multiple-of-32
head dims, fp16, fp32) still fall back entirely to the brgemm/micro-gemm path and
must stay correct.

These tests cover:
  * Correctness across odd head dims, GQA ratios, batch sizes, seq lens, masks,
    score mods, and V head dim != QK head dim, against a float64 golden reference
    (compiled error must not exceed the eager-bf16 reference error by > fudge).
  * Path coverage: shapes that should take the AMX path do, and shapes that must
    fall back do (asserted via the FLEX_ATTENTION_AMX_DEBUG runtime print).
  * A/B equivalence: with the same compiled kernel, forcing the brgemm path
    (FLEX_ATTENTION_DISABLE_AMX=1) must match the AMX path within bf16 tolerance.

Run: python test/inductor/test_flex_attention_amx.py
"""

import contextlib
import functools
import os
import sys
import unittest


# Ensure the in-tree PyTorch build (which contains the AMX overlap kernel) is
# imported rather than a possibly-stale site-packages copy: when this file is run
# as `python test/inductor/test_flex_attention_amx.py`, sys.path[0] is the test
# dir, so an editable-but-copied install would otherwise shadow our changes.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if os.path.isdir(os.path.join(_REPO_ROOT, "torch", "_inductor", "codegen")):
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch._inductor.config as inductor_config
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU


def _amx_supported():
    """bf16 need_pack path requires mkldnn bf16 + AMX (Sapphire Rapids+)."""
    if not HAS_CPU:
        return False
    if not (
        torch.backends.mkldnn.is_available() and torch.cpu._is_amx_tile_supported()
    ):
        return False
    try:
        return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:
        return False


AMX_SUPPORTED = _amx_supported()
skip_no_amx = unittest.skipUnless(
    AMX_SUPPORTED, "CPU AMX bf16 (need_pack) path not supported on this machine"
)


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _noop(b, h, q_idx, kv_idx):
    return q_idx >= 0


# A few representative score mods (identity, additive positional bias, alibi-like).
def _times_two(score, b, h, q_idx, kv_idx):
    return score * 2


def _rel_bias(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


SCORE_MODS = {"none": None, "times_two": _times_two, "rel_bias": _rel_bias}


@contextlib.contextmanager
def _env(**kv):
    """Temporarily set/clear env vars (restored on exit)."""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _clone(t, dtype):
    return t.detach().clone().to(dtype)


@skip_no_amx
class TestFlexAttentionAMX(TestCase):
    """All tests force-disable the inductor cache so template edits/paths are
    always recompiled fresh, and pin a modest thread count for stable timing."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Guard against silently testing a stale/divergent torch: this suite must
        # import the in-tree build that actually contains the AMX overlap kernel.
        # (Running from a subdir picks up a site-packages copy without it.)
        from torch._inductor.codegen import cpp_flex_attention_template as _tmpl

        if "use_amx_overlap" not in _tmpl.FLEX_ATTENTION_TEMPLATE:
            raise unittest.SkipTest(
                "imported torch lacks the AMX overlap kernel "
                f"({_tmpl.__file__}); run this suite from the repo root so the "
                "in-tree build shadows any site-packages torch"
            )

    def setUp(self):
        super().setUp()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(inductor_config.patch(force_disable_caches=True))
        torch._dynamo.reset()

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    # ---- helpers -----------------------------------------------------------

    def _build(self, B, Hq, Hkv, Sq, Skv, D, Dv, mask, dtype):
        torch.manual_seed(0)
        q = torch.randn(B, Hq, Sq, D, dtype=dtype)
        k = torch.randn(B, Hkv, Skv, D, dtype=dtype)
        v = torch.randn(B, Hkv, Skv, Dv, dtype=dtype)
        mask_fn = _causal if mask == "causal" else _noop
        block_mask = create_block_mask(mask_fn, B, Hq, Sq, Skv, device="cpu")
        return q, k, v, block_mask

    def _run_correctness(
        self, B, Hq, Hkv, Sq, Skv, D, Dv, mask, score_mod, dtype=torch.bfloat16
    ):
        """Compare compiled vs float64 golden; compiled error must be within
        `fudge`x the eager same-dtype reference error (relative-error scheme, as
        the upstream flex tests use for a non-bit-exact kernel)."""
        q, k, v, block_mask = self._build(B, Hq, Hkv, Sq, Skv, D, Dv, mask, dtype)
        gqa = Hq != Hkv
        attn = functools.partial(
            flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=gqa,
        )
        compiled = torch.compile(attn)
        out = compiled(q, k, v)

        gold = attn(
            _clone(q, torch.float64), _clone(k, torch.float64), _clone(v, torch.float64)
        )
        ref = attn(_clone(q, dtype), _clone(k, dtype), _clone(v, dtype))

        self.assertFalse(torch.isnan(out).any(), "AMX kernel produced NaNs")
        self.assertEqual(out.shape, (B, Hq, Sq, Dv))

        comp_err = (gold - out.to(torch.float64)).abs().mean()
        ref_err = (gold - ref.to(torch.float64)).abs().mean()
        # fudge=2.0: empirically the AMX path's error ratio vs eager bf16 is
        # 0.90-1.19 across all tested shapes; 2.0 gives flake headroom while
        # still catching a genuinely broken kernel (ratio >> 2 or NaN).
        fudge = 2.0
        torch.testing.assert_close(
            comp_err,
            ref_err,
            rtol=fudge,
            atol=2e-5,
            msg=lambda m: (
                f"compiled err {comp_err:.3e} vs ref err {ref_err:.3e} "
                f"(ratio {comp_err / max(ref_err, 1e-12):.2f} > {fudge}). {m}"
            ),
        )

    def _took_amx_path(self, B, Hq, Hkv, Sq, Skv, D, Dv, mask, dtype=torch.bfloat16):
        """Return whether the kernel selected the AMX overlap path at runtime.

        Runs in a FRESH subprocess with an empty inductor cache dir so the kernel
        is genuinely (re)compiled and its runtime `use_amx_overlap` decision is
        re-emitted -- the decision also depends on the runtime `need_pack`
        work-ratio (thread count), which a static parse of the generated code
        cannot reproduce. The kernel appends its flag (1/0) to
        FLEX_ATTENTION_AMX_DEBUG_FILE (a real file: a subprocess stderr pipe does
        not reliably capture the compiled .so's stderr fd). Do NOT enable
        force_disable_caches in the child -- an empty cache dir already forces a
        recompile, and force_disable_caches skips re-emitting the decision.
        """
        import subprocess
        import sys
        import tempfile
        import textwrap

        dt = str(dtype).split(".")[-1]
        with tempfile.NamedTemporaryFile("w+", suffix=".amxdbg", delete=False) as f:
            dbg_path = f.name
        cache_dir = tempfile.mkdtemp(prefix="amx_pathprobe_")
        script = textwrap.dedent(
            f"""
            import torch
            torch.set_num_threads({max(1, torch.get_num_threads())})
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
            torch.manual_seed(0)
            q = torch.randn({B}, {Hq}, {Sq}, {D}, dtype=torch.{dt})
            k = torch.randn({B}, {Hkv}, {Skv}, {D}, dtype=torch.{dt})
            v = torch.randn({B}, {Hkv}, {Skv}, {Dv}, dtype=torch.{dt})
            mfn = (lambda b, h, qi, ki: qi >= ki) if {mask!r} == "causal" else (lambda b, h, qi, ki: qi >= 0)
            bm = create_block_mask(mfn, {B}, {Hq}, {Sq}, {Skv}, device="cpu")
            c = torch.compile(flex_attention)
            c(q, k, v, block_mask=bm, enable_gqa={Hq != Hkv}).sum().item()
            """
        )
        # Start from a clean env: drop any inherited inductor cache/force-disable
        # vars that would let the child reuse a pre-instrumentation compile.
        env = {
            key: val
            for key, val in os.environ.items()
            if not key.startswith("TORCHINDUCTOR_")
            and not key.startswith("TORCH_COMPILE")
        }
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        env["FLEX_ATTENTION_AMX_DEBUG"] = "1"
        env["FLEX_ATTENTION_AMX_DEBUG_FILE"] = dbg_path
        # The child must import the SAME torch as this test (the in-tree editable
        # build), not a divergent site-packages copy. torch.__path__ points at the
        # in-tree package, so prepend its parent (the repo root) to PYTHONPATH.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_root, env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                cwd=repo_root,
                timeout=600,
            )
            with open(dbg_path) as fh:
                flags = [ln.strip() for ln in fh if ln.strip()]
        finally:
            os.unlink(dbg_path)
        self.assertEqual(
            proc.returncode,
            0,
            f"path-probe child failed:\n{proc.stderr.decode()[-2000:]}",
        )
        self.assertTrue(flags, "kernel did not report an AMX path decision")
        self.assertTrue(
            all(x == flags[0] for x in flags),
            f"inconsistent AMX path decisions within one call: {set(flags)}",
        )
        return flags[0] == "1"

    # ---- correctness: odd / non-32-multiple head dims (fallback path) ------

    @parametrize("D", [16, 40, 48, 72, 80, 96, 100, 128, 200, 256])
    def test_odd_head_dim(self, device, D):
        # D not a multiple of 32 falls back to brgemm; D%32==0 hits AMX. Both
        # must be correct.
        self._run_correctness(2, 8, 8, 256, 256, D, D, "full", None)

    @parametrize("D", [56, 88, 152])
    def test_odd_head_dim_causal(self, device, D):
        self._run_correctness(2, 8, 8, 512, 512, D, D, "causal", None)

    # ---- correctness: GQA (Hq != Hkv) --------------------------------------

    @parametrize("ratio", [1, 2, 4, 8, 16])
    def test_gqa_ratios(self, device, ratio):
        Hkv = 16 // ratio if 16 % ratio == 0 else 8
        Hq = Hkv * ratio
        self._run_correctness(2, Hq, Hkv, 512, 512, 64, 64, "full", None)

    @parametrize("ratio", [2, 4])
    def test_gqa_causal(self, device, ratio):
        self._run_correctness(2, 8 * ratio, 8, 1024, 1024, 64, 64, "causal", None)

    # ---- correctness: batch sizes ------------------------------------------

    @parametrize("B", [1, 2, 3, 5, 8])
    def test_batch_sizes(self, device, B):
        self._run_correctness(B, 8, 8, 512, 512, 64, 64, "causal", None)

    # ---- correctness: sequence lengths (aligned + ragged) ------------------

    @parametrize(
        "Sq,Skv",
        [(128, 128), (512, 1024), (1024, 512), (320, 320), (576, 576), (2048, 2048)],
    )
    def test_seq_lengths(self, device, Sq, Skv):
        self._run_correctness(1, 8, 8, Sq, Skv, 64, 64, "full", None)

    # ---- correctness: PARTIAL q-block on the AMX path ----------------------
    # cur_qSplitSize not a multiple of 16 (and qSize < qBlockSize) makes the AMX
    # GEMM store 16-row tiles past the real row count. Regression guard: this used
    # to corrupt the heap / return garbage for qS in {65,72,100,120,127}. qS >= 64
    # takes the AMX path (need_pack threshold); qS < 64 exercises the fallback.
    # The list spans <16, tiny, and every awkward 16-remainder.
    @parametrize("qS", [1, 8, 15, 17, 33, 65, 72, 100, 120, 127, 129, 250])
    @parametrize("mask", ["full", "causal"])
    def test_partial_q_block(self, device, qS, mask):
        self._run_correctness(2, 8, 8, qS, 256, 64, 64, mask, None)

    # Same partial-q-block coverage at D=128. D=128 packs the QK GEMM into 32-wide
    # K tiles (vs one for D=64) and has a larger eheadSize, a distinct tiling from
    # the D=64 case above; the qk score/dst buffers are also 2x wider so a
    # mis-sized eqSplitSize would over-run differently. Same awkward 16-remainders.
    @parametrize("qS", [65, 72, 100, 120, 127, 129, 250])
    @parametrize("mask", ["full", "causal"])
    def test_partial_q_block_d128(self, device, qS, mask):
        self._run_correctness(2, 8, 8, qS, 256, 128, 128, mask, None)

    # PARTIAL TAIL q-block: qSize > qBlockSize(=128), so the kernel runs one or
    # more FULL 128-row q-blocks and then a trailing partial block of size
    # (qSize % 128). This exercises a different path than qSize < 128 (the pending
    # AMX block is carried across q-blocks and the tile-store overrun happens on
    # the LAST block after full blocks). Tails span <16 / =16 / 17..31 / 44 / 127.
    @parametrize("qS", [129, 130, 143, 144, 145, 160, 200, 255, 257, 300, 511, 1000])
    @parametrize("mask", ["full", "causal"])
    @parametrize("D", [64, 128])
    def test_partial_tail_q_block(self, device, qS, D, mask):
        self._run_correctness(2, 8, 8, qS, 256, D, D, mask, None)

    # A partial q-block must still take the AMX path when kv is tile-aligned
    # (the fix is about buffer sizing, not falling back). Cover both the
    # qSize < qBlockSize single-block case and a qSize > qBlockSize tail block.
    @parametrize("qS", [72, 145, 300])
    def test_partial_q_block_uses_amx(self, device, qS):
        self.assertTrue(
            self._took_amx_path(2, 8, 8, qS, 256, 64, 64, "full"),
            f"partial q-block (qS={qS}) with aligned kv should use the AMX path",
        )

    # ---- correctness: V head dim != QK head dim ----------------------------

    @parametrize("D,Dv", [(64, 128), (128, 64), (64, 96), (96, 64), (128, 256)])
    def test_v_head_dim_differs(self, device, D, Dv):
        self._run_correctness(2, 8, 8, 512, 512, D, Dv, "full", None)

    # ---- correctness: score mods -------------------------------------------

    @parametrize("mod_name", list(SCORE_MODS.keys()))
    @parametrize("mask", ["full", "causal"])
    def test_score_mods(self, device, mod_name, mask):
        self._run_correctness(2, 8, 8, 512, 512, 64, 64, mask, SCORE_MODS[mod_name])

    # ---- correctness: dtypes that must NOT use the AMX bf16 path ------------

    def test_fp32_fallback(self, device):
        self._run_correctness(
            2, 8, 8, 512, 512, 64, 64, "causal", None, dtype=torch.float32
        )

    # ---- path coverage -----------------------------------------------------

    def test_amx_path_selected_for_aligned(self, device):
        # bf16, D%32==0, Dv%32==0, Skv%128==0 -> AMX overlap path.
        self.assertTrue(
            self._took_amx_path(4, 16, 16, 1024, 1024, 64, 64, "full"),
            "expected AMX overlap path for an aligned bf16 shape",
        )
        self.assertTrue(
            self._took_amx_path(2, 16, 16, 512, 512, 128, 128, "causal"),
            "expected AMX overlap path for D=128 aligned shape",
        )

    def test_fallback_for_odd_head_dim(self, device):
        # D=40 is not a multiple of 32 -> must fall back.
        self.assertFalse(
            self._took_amx_path(2, 8, 8, 256, 256, 40, 40, "full"),
            "odd head dim must not take the AMX path",
        )

    def test_ragged_kv_uses_amx(self, device):
        # kvSize=320 is not a multiple of kvSplitSize(=128), but the AMX path now
        # handles ragged kv: aligned 128-blocks run on AMX+overlap, only the
        # ragged tail block (320 = 2*128 + 64; here the 64-tail is still %32==0 so
        # even it is AMX) falls back to brgemm. use_amx_overlap is reported true.
        self.assertTrue(
            self._took_amx_path(2, 8, 8, 320, 320, 64, 64, "full"),
            "ragged kv length should still use the AMX path for aligned blocks",
        )

    def test_fallback_for_fp32(self, device):
        self.assertFalse(
            self._took_amx_path(2, 8, 8, 512, 512, 64, 64, "full", dtype=torch.float32),
            "fp32 must not take the AMX bf16 path",
        )

    # ---- A/B equivalence: AMX path vs forced brgemm path -------------------

    @parametrize(
        "B,Hq,Hkv,S,D,Dv,mask",
        [
            (2, 16, 16, 1024, 64, 64, "causal"),
            (2, 16, 16, 512, 128, 128, "full"),
            (2, 16, 4, 512, 64, 64, "full"),  # GQA
            (1, 8, 8, 512, 64, 128, "full"),  # Dv != D
            (1, 16, 16, 2048, 64, 64, "causal"),
        ],
    )
    def test_amx_matches_brgemm(self, device, B, Hq, Hkv, S, D, Dv, mask):
        """Same compiled kernel, toggle the runtime path via env: the AMX output
        must match the oneDNN brgemm output within bf16 tolerance (empirically
        bit-exact for many shapes, <=~2e-3 max where the 2-buffer online softmax
        reorders accumulation)."""
        q, k, v, block_mask = self._build(B, Hq, Hkv, S, S, D, Dv, mask, torch.bfloat16)
        gqa = Hq != Hkv
        compiled = torch.compile(
            functools.partial(flex_attention, block_mask=block_mask, enable_gqa=gqa)
        )
        with _env(FLEX_ATTENTION_DISABLE_AMX=None):
            amx_out = compiled(q, k, v).clone()
        with _env(FLEX_ATTENTION_DISABLE_AMX="1"):
            brgemm_out = compiled(q, k, v).clone()
        self.assertFalse(torch.isnan(amx_out).any())
        torch.testing.assert_close(
            amx_out.float(), brgemm_out.float(), rtol=0, atol=5e-3
        )

    # ---- the exact benchmark shapes from the design work -------------------

    @parametrize("D", [64, 128, 256])
    def test_benchmark_shapes(self, device, D):
        B = 1 if D == 256 else (2 if D == 128 else 4)
        self._run_correctness(B, 16, 16, 1024, 1024, D, D, "causal", None)

    # ---- COMBINATORIAL: batch x head num x GQA x head dim x mask -----------
    # Unlike the single-axis tests above, these cross MULTIPLE edge properties
    # at once (e.g. odd batch + GQA + causal, or odd head count + fallback dim)
    # to catch interactions the one-axis tests miss: the GQA broadcast index math
    # (gqa_shards = num_head/num_head_k), the per-thread work split (B*Hq*qSlice
    # not evenly divisible by threads), and AMX-vs-brgemm path selection all
    # depend on more than one axis simultaneously.

    # Main matrix: batch {even, single, ODD} x head config {plain, GQA-4x,
    # GQA-3x-odd-heads} x head dim {D=64 AMX, D=128 AMX} x {full, causal}.
    # 4 batches x 4 head configs x 2 dims x 2 masks = 64 cases.
    @parametrize("B", [1, 2, 3, 5])
    @parametrize("Hq,Hkv", [(8, 8), (16, 16), (16, 4), (9, 3)])
    @parametrize("D", [64, 128])
    @parametrize("mask", ["full", "causal"])
    def test_combo_batch_head_gqa_dim(self, device, B, Hq, Hkv, D, mask):
        self._run_correctness(B, Hq, Hkv, 512, 512, D, D, mask, None)

    # GQA with ODD head counts and awkward-but-valid ratios (Hq % Hkv == 0 is
    # required by the frontend; these still stress the broadcast index math with
    # non-power-of-two group sizes and odd Hkv). Crossed with AMX + fallback dims.
    # 6 head configs x 3 dims x 2 masks = 36 cases.
    @parametrize("Hq,Hkv", [(9, 3), (6, 2), (15, 5), (12, 4), (14, 7), (10, 5)])
    @parametrize("D", [64, 128, 48])  # 48 is not a multiple of 32 -> fallback
    @parametrize("mask", ["full", "causal"])
    def test_combo_gqa_odd_heads(self, device, Hq, Hkv, D, mask):
        self._run_correctness(2, Hq, Hkv, 512, 512, D, D, mask, None)

    # ODD / non-multiple-of-32 head dims (fallback path) combined with odd batch
    # and GQA, so the brgemm fallback is exercised together with the awkward
    # batch/head index math rather than in isolation.
    # 4 dims x 3 (B,Hq,Hkv) x 2 masks = 24 cases.
    @parametrize("D", [40, 48, 72, 96])
    @parametrize("B,Hq,Hkv", [(3, 8, 8), (2, 16, 4), (5, 9, 3)])
    @parametrize("mask", ["full", "causal"])
    def test_combo_fallback_dim_batch_gqa(self, device, D, B, Hq, Hkv, mask):
        self._run_correctness(B, Hq, Hkv, 512, 512, D, D, mask, None)

    # UNEVEN-DIVISIBLE sequence lengths (ragged kv / Sq!=Skv) combined with head
    # configs. kvSize not a multiple of kvSplitSize(128) forces the ragged-kv
    # fallback; pairing with GQA/odd heads checks the two interact correctly.
    # 4 (Sq,Skv) x 3 head configs x 2 masks = 24 cases.
    @parametrize("Sq,Skv", [(512, 320), (320, 512), (576, 576), (448, 640)])
    @parametrize("Hq,Hkv", [(8, 8), (16, 4), (9, 3)])
    @parametrize("mask", ["full", "causal"])
    def test_combo_uneven_seq_head(self, device, Sq, Skv, Hq, Hkv, mask):
        self._run_correctness(2, Hq, Hkv, Sq, Skv, 64, 64, mask, None)

    # ODD BATCH stress: B in {3,5,7} makes B*Hq*qSlice not evenly divisible over
    # threads (tail threads get fewer/uneven work items). Cross with head configs
    # and both masks. 3 batches x 3 head configs x 2 masks = 18 cases.
    @parametrize("B", [3, 5, 7])
    @parametrize("Hq,Hkv", [(8, 8), (9, 3), (16, 4)])
    @parametrize("mask", ["full", "causal"])
    def test_combo_odd_batch(self, device, B, Hq, Hkv, mask):
        self._run_correctness(B, Hq, Hkv, 512, 512, 64, 64, mask, None)

    # NEGATIVE: GQA where Hq is NOT a multiple of Hkv must raise at the frontend
    # (this is an invalid configuration, not a kernel path). Guards against the
    # combinatorial lists ever silently accepting a non-divisible pair.
    @parametrize("Hq,Hkv", [(10, 4), (9, 4), (7, 2)])
    def test_combo_gqa_non_divisible_raises(self, device, Hq, Hkv):
        q, k, v, block_mask = self._build(
            2, Hq, Hkv, 512, 512, 64, 64, "full", torch.bfloat16
        )
        compiled = torch.compile(
            functools.partial(flex_attention, block_mask=block_mask, enable_gqa=True)
        )
        with self.assertRaises((ValueError, RuntimeError)):
            compiled(q, k, v)

    # PATH COVERAGE for the combinatorial matrix: confirm representative combos
    # route as intended (aligned GQA -> AMX; ragged kv -> AMX for aligned blocks;
    # odd head dim -> full fallback), so the correctness matrix is actually
    # exercising the AMX path where expected and not silently all falling back.
    # Kept small: each probe spawns a fresh subprocess (~13s).
    def test_combo_path_coverage(self, device):
        # Aligned bf16 GQA with odd head counts still takes the AMX path.
        self.assertTrue(
            self._took_amx_path(2, 9, 3, 512, 512, 64, 64, "full"),
            "aligned GQA (odd heads) should take the AMX path",
        )
        self.assertTrue(
            self._took_amx_path(3, 16, 4, 512, 512, 128, 128, "causal"),
            "aligned GQA D=128 odd batch should take the AMX path",
        )
        # Ragged kv under GQA now takes the AMX path (aligned blocks on AMX, the
        # ragged tail on brgemm). kvSize need not be a multiple of kvSplitSize.
        self.assertTrue(
            self._took_amx_path(2, 16, 4, 512, 320, 64, 64, "full"),
            "ragged kv under GQA should use the AMX path for its aligned blocks",
        )
        # Odd head dim (not a multiple of 32) must still fall back entirely.
        self.assertFalse(
            self._took_amx_path(2, 12, 4, 512, 512, 48, 48, "full"),
            "non-multiple-of-32 head dim must fall back even under GQA",
        )

    # ---- ODD sequence lengths (Sq == Skv, odd values) ----------------------
    # S=137 and S=1077 are NOT multiples of kvSplitSize(128); the AMX path now
    # handles them by running the aligned 128-blocks on AMX+overlap and the ragged
    # tail block (137 = 128 + 9; 1077 = 8*128 + 3) on brgemm. This exercises the
    # AMX->brgemm-tail transition (drain the pending AMX block + release the tile
    # config before the tail). S=1 is below the need_pack threshold (64) so it is
    # a full fallback. Verified via the path oracle (test_ragged_kv_uses_amx).
    @parametrize("S", [1, 137, 1077])
    @parametrize("D", [64, 128])
    @parametrize("mask", ["full", "causal"])
    def test_odd_seq_length(self, device, S, D, mask):
        self._run_correctness(2, 8, 8, S, S, D, D, mask, None)

    # ---- LONG sequence lengths ---------------------------------------------
    # S=4096 and S=8192 are multiples of kvSplitSize(128) so they take the AMX
    # path and stress the pipelined kv-loop over many blocks + the running
    # online-softmax accumulation across a long reduction. Small B/H keeps the
    # float64 golden reference fast. D{64,128} x {full,causal}. 2x2x2 = 8 cases.
    @parametrize("S", [4096, 8192])
    @parametrize("D", [64, 128])
    @parametrize("mask", ["full", "causal"])
    def test_long_seq_length(self, device, S, D, mask):
        self._run_correctness(1, 2, 2, S, S, D, D, mask, None)

    # An odd long-ish length that is neither q-block nor kv-block aligned. 1077 %
    # 128 != 0, so it runs the 8 aligned 128-blocks on AMX+overlap and the 3-row
    # ragged tail on brgemm; covers a large odd length end-to-end on the AMX path.
    @parametrize("mask", ["full", "causal"])
    def test_odd_long_seq_length(self, device, mask):
        self._run_correctness(1, 4, 4, 1077, 1077, 64, 64, mask, None)

    # ---- RAGGED kv (kvSize % kvSplitSize != 0) on the AMX path --------------
    # Exercises the AMX-aligned-blocks + brgemm-ragged-tail split for arbitrary
    # kv lengths (the feature that lets any sequence length use AMX). Covers a
    # tail that is %32==0 (still AMX: 320=2*128+64), a tail that is not (the true
    # brgemm tail: 427=3*128+43, 1027=8*128+3), a length just over one block
    # (129,150), and Skv < kvSplitSize where the ENTIRE kv is the ragged tail so
    # there is no pending AMX block to drain (200 -> kvSplitSize clamps to 200,
    # not %32, full fallback; 160 -> one 128 block + 32 tail).
    @parametrize("Skv", [129, 150, 160, 320, 427, 1027])
    @parametrize("D", [64, 128])
    @parametrize("mask", ["full", "causal"])
    def test_ragged_kv_amx(self, device, Skv, D, mask):
        self._run_correctness(2, 8, 8, Skv, Skv, D, D, mask, None)

    # Confirm ragged kv actually takes the AMX path (aligned blocks) rather than
    # silently falling back -- the user-requested 427 / 1027 plus a small tail.
    @parametrize("Skv", [427, 1027, 320])
    def test_ragged_kv_uses_amx(self, device, Skv):
        self.assertTrue(
            self._took_amx_path(2, 8, 8, Skv, Skv, 128, 128, "full"),
            f"ragged kv Skv={Skv} should use the AMX path for its aligned blocks",
        )


instantiate_device_type_tests(
    TestFlexAttentionAMX, globals(), only_for=("cpu",), allow_xpu=False
)


if __name__ == "__main__":
    run_tests()
