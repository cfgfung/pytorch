# mypy: allow-untyped-defs
"""AMX/AVX-512 interleaved GEMM helpers for the CPU FlexAttention template.

AMX tile registers only accept literal-immediate indices, so the two supported
row counts (16 and 32) are separate specializations with hardcoded tile indices
that follow ``AMXState``'s slot order (C tiles, then A, then B).
"""

# The C++ is a Jinja template only to substitute the kernel-name prefix so the
# symbols do not collide across multiple compiled kernels in one process.
FLEX_ATTENTION_AMX_HELPERS = r"""
// Zero-padded the rows up to a multiple of 16 so the AMX Q@K^T can read whole 16-row A tiles. 
// Also fold the scale into the Q.
template <typename scalar_t>
inline void {{kernel_name}}_amx_scale_q(
    const scalar_t* q_ptr,
    scalar_t* out_ptr,
    float scale,
    int64_t rows,
    int64_t prows,
    int64_t cols,
    int64_t pcols,
    int64_t ldi) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t vec_size = Vec::size();
  auto fscale = at::vec::Vectorized<float>(scale);
  for (int64_t r = 0; r < rows; ++r) {
    const scalar_t* src = q_ptr + r * ldi;
    scalar_t* dst = out_ptr + r * pcols;
    int64_t c = 0;
    if constexpr (c10::is_reduced_floating_point_v<scalar_t>) {
      for (; c < vec_size * (cols / vec_size); c += vec_size) {
        auto [v0, v1] = at::vec::convert_to_float<scalar_t>(Vec::loadu(src + c));
        at::vec::convert_from_float<scalar_t>(v0 * fscale, v1 * fscale).store(dst + c);
      }
      for (; c < cols; ++c) dst[c] = static_cast<scalar_t>(static_cast<float>(src[c]) * scale);
    } else {
      auto vscale = Vec(static_cast<scalar_t>(scale));
      for (; c < vec_size * (cols / vec_size); c += vec_size) {
        (Vec::loadu(src + c) * vscale).store(dst + c);
      }
      for (; c < cols; ++c) dst[c] = src[c] * static_cast<scalar_t>(scale);
    }
    for (int64_t p = cols; p < pcols; ++p) dst[p] = static_cast<scalar_t>(0);
  }
  for (int64_t r = rows; r < prows; ++r) {
    scalar_t* dst = out_ptr + r * pcols;
    for (int64_t p = 0; p < pcols; ++p) dst[p] = static_cast<scalar_t>(0);
  }
}

// AMX bf16 accumulator block. C[NROWS,32] (+)= A[NROWS,K] @ Bp[K,32] (VNNI2).
// Tile indices match AMXState slot order. cb() runs after each K-step so the
// caller can interleave AVX softmax work with the AMX tdp stream.
template <bool accum, typename CB>
inline void {{kernel_name}}_amx_block32(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  auto load_cfg = [](const amx_tilecfg& c) { _tile_loadconfig(&c); };
  amx_state.configure(16, 64, 2, 2, load_cfg);
  if constexpr (accum) {
    _tile_loadd(0, C, ldc * sizeof(float));
    _tile_loadd(1, C + 16, ldc * sizeof(float));
    _tile_loadd(2, C + 16 * ldc, ldc * sizeof(float));
    _tile_loadd(3, C + 16 * ldc + 16, ldc * sizeof(float));
  } else {
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  }
  for (int64_t k = 0; k < K; k += 32) {
    const {{amx_t}}* Ak = A + k;
    const {{amx_t}}* Bk = B + k * ldb;
    int64_t kn = k + 32;
    if (kn < K) {
      const {{amx_t}}* Bp = B + kn * ldb;
      _mm_prefetch(reinterpret_cast<const char*>(Bp), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 64), _MM_HINT_T0);
    }
    _tile_loadd(4, Ak, lda * sizeof({{amx_t}}));
    _tile_loadd(6, Bk, ldb * 2 * sizeof({{amx_t}}));
    _tile_loadd(7, Bk + 32, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(0, 4, 6);
    _tile_loadd(5, Ak + 16 * lda, lda * sizeof({{amx_t}}));
    _tile_dpbf16ps(1, 4, 7);
    _tile_dpbf16ps(2, 5, 6);
    _tile_dpbf16ps(3, 5, 7);
    cb();
  }
  _tile_stored(0, C, ldc * sizeof(float));
  _tile_stored(1, C + 16, ldc * sizeof(float));
  _tile_stored(2, C + 16 * ldc, ldc * sizeof(float));
  _tile_stored(3, C + 16 * ldc + 16, ldc * sizeof(float));
}

template <bool accum, typename CB>
inline void {{kernel_name}}_amx_block16(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t K, int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  auto load_cfg = [](const amx_tilecfg& c) { _tile_loadconfig(&c); };
  amx_state.configure(16, 64, 1, 2, load_cfg);
  if constexpr (accum) {
    _tile_loadd(0, C, ldc * sizeof(float));
    _tile_loadd(1, C + 16, ldc * sizeof(float));
  } else {
    _tile_zero(0); _tile_zero(1);
  }
  for (int64_t k = 0; k < K; k += 32) {
    const {{amx_t}}* Bk = B + k * ldb;
    int64_t kn = k + 32;
    if (kn < K) {
      const {{amx_t}}* Bp = B + kn * ldb;
      _mm_prefetch(reinterpret_cast<const char*>(Bp), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 32), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(Bp + 64), _MM_HINT_T0);
    }
    _tile_loadd(2, A + k, lda * sizeof({{amx_t}}));
    _tile_loadd(3, Bk, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(0, 2, 3);
    _tile_loadd(4, Bk + 32, ldb * 2 * sizeof({{amx_t}}));
    _tile_dpbf16ps(1, 2, 4);
    cb();
  }
  _tile_stored(0, C, ldc * sizeof(float));
  _tile_stored(1, C + 16, ldc * sizeof(float));
}

// Main GEMM function - C[M,N] (+)= A[M,K] rowmajor(lda) @ Bp[K,N] VNNI2(ldb).
template <bool accum, typename CB>
inline void {{kernel_name}}_amx_gemm_cb(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, CB cb) {
  for (int64_t m = 0; m < M; m += 32) {
    // 32-row (2 M-tiles) panel when >=32 rows remain, else a 16-row panel that
    // covers the final 1..32 rows (rounding the last <16 remainder up to 16).
    int64_t nrows = (M - m) > 16 ? 32 : 16;
    for (int64_t n = 0; n < N; n += 32) {
      const {{amx_t}}* Ablk = A + m * lda;
      const {{amx_t}}* Bblk = B + n * 2;
      float* Cblk = C + m * ldc + n;
      if (nrows == 32)
        {{kernel_name}}_amx_block32<accum>(amx_state, Ablk, Bblk, Cblk, K, lda, ldb, ldc, cb);
      else
        {{kernel_name}}_amx_block16<accum>(amx_state, Ablk, Bblk, Cblk, K, lda, ldb, ldc, cb);
    }
  }
}

template <bool accum>
inline void {{kernel_name}}_amx_gemm(
    AMXState& amx_state,
    const {{amx_t}}* A, const {{amx_t}}* B, float* C,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc) {
  {{kernel_name}}_amx_gemm_cb<accum>(amx_state, A, B, C, M, N, K, lda, ldb, ldc, []() {});
}

// One row of online softmax for the AMX bf16 path (scale already folded into Q).
// This is the AVX-512 work interleaved with the next block's AMX Q@K^T GEMM.
template <typename scalar_t>
inline void {{kernel_name}}_amx_online_softmax_row(
    float* qk_row,
    scalar_t* p_row,
    int64_t cur_kvSplitSize,
    float& row_max,
    float& row_sum,
    float* dst_row,
    int64_t headSize_v,
    bool first_block) {
  using Vec = at::vec::Vectorized<float>;
  float block_max = -std::numeric_limits<float>::infinity();
  {{kernel_name}}_mul_reduce_max_fusion_kernel(
      qk_row, static_cast<float>(1), cur_kvSplitSize, qk_row, block_max);
  float new_max = row_max > block_max ? row_max : block_max;
  if (new_max == -std::numeric_limits<float>::infinity()) {
    {{kernel_name}}_fill_stub(p_row, static_cast<scalar_t>(0), cur_kvSplitSize);
  } else {
    float block_sum = new_max;
    {{kernel_name}}_exp_reduce_sum_fusion_kernel(
        qk_row, cur_kvSplitSize, p_row, block_sum);
    float exp_tmp = std::exp(row_max - new_max);
    row_sum = block_sum + exp_tmp * row_sum;
    if (!first_block) {
      at::vec::map<float>(
          [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
          dst_row, dst_row, headSize_v);
    }
  }
  row_max = new_max;
  if (cur_kvSplitSize % 2 != 0) {
    p_row[cur_kvSplitSize] = static_cast<scalar_t>(0);
  }
}
"""


def codegen_flex_attention_amx_helpers(kernel_name: str) -> str:
    """Render the AMX/AVX interleaving helpers for the given kernel-name prefix.

    ``amx_t`` is ``uint16_t`` -- the packed K/V and pre-scaled Q buffers are
    reinterpreted to 16-bit at the call sites (both BFloat16 and Half are 2-byte),
    matching what ``pack_vnni2`` and ``_tile_dpbf16ps`` consume.
    """
    from .common import KernelTemplate

    return KernelTemplate._template_from_string(FLEX_ATTENTION_AMX_HELPERS).render(
        dict(kernel_name=kernel_name, amx_t="uint16_t")
    )
