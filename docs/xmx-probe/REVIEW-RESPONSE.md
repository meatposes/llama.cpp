# Review response: XMX Q2_0 GEMM NO-GO

Reviewer verdict on `REVIEW-REQUEST.md`. Read the kernels and the peak bench directly.

## Verdict (two parts)

1. **NO-GO for the int8-fused kernel: AGREE, now airtight.** Measured: oneDNN f16 GEMM is 10.9x
   faster on the matmul alone (0.732 ms vs 8.0 ms). The plateau is Q8_1's per-32-K scaling stalling
   the pipeline - a real architectural wall, not an unoptimized kernel.
2. **BUT the broader "fused XMX GEMM isn't worth it" is TOO STRONG.** A different framing -
   **fp16-fused with in-register weight dequant, no activation quantization** - sidesteps the exact
   wall that killed int8. Lower ceiling (~1.3x, not 1.6x) but actually reachable. This was not
   tried and is not on the "already tried" list. Details below.

## Fact-check of the claims

- **int8 358 / fp16 183 TOPS, 1.95x ratio:** sound as a *ratio*. Caveat on the absolute peak:
  `xmx_peak_int8_vs_fp16.cpp` runs a serial `mad(tC,tA,tB,tC)` chain with tA/tB loop-invariant and
  a single tC dependency per work-item. The 16384 sub-groups hide DPAS latency, so it's a
  reasonable throughput proxy, but treat 358 as "peak-ish upper bound," not exact. The 1.95x ratio
  is the robust, load-bearing number and it is correct - int8 DPAS is 2x fp16 on Xe2.
- **11.3 TOPS plateau, MSTRIP>4 spills:** confirmed by reading `q2_tiled_inreg_scale.cpp`. Four
  persistent `joint_matrix<float,accumulator>` per work-item + tA/tB/tC is already heavy; MSTRIP 8/16
  spilling is expected.
- **The core stall is real and correctly identified.** In the inner loop, immediately after
  `joint_matrix_mad` the code does `get_wi_data(tC)` + `get_coord` + scale into facc. That read
  forces the DPAS result to be consumed before the next mad in the same mi-chain, and with only
  MSTRIP=4 independent chains there is not enough ILP to hide DPAS latency behind the scalar scale.
  This serializes DPAS with scalar work every 32-K. Diagnosis is right.
- **Secondary inefficiency not called out:** the activation load
  `joint_matrix_load(tA, aptr, Astride=5760)` is a strided/uncoalesced global read (8 rows, 32B
  each, 5760B apart) redone every sub-block. Minor vs the stall, but real; SLM-staging A per
  m-strip or a transposed activation layout would help a properly-tuned version.
- **~8x vs oneDNN: MEASURED, actually ~11x.** oneDNN f16 matmul at K=5120 N=17408 M=512 =
  **0.732 ms/iter, 124.7 TOPS** (`onednn_f16_gemm_bench.cpp`, fpmath_mode::f16, sycl_interop,
  level_zero:1). Fused int8 kernel = 8.0 ms -> **oneDNN is 10.9x faster on the GEMM alone**, before
  even adding the dequant the fused kernel is meant to save. The ~8x estimate was conservative.
  oneDNN achieves 68% of the 183 fp16 peak - that 124.7 TOPS is the realistic bar for any fp16-fused
  kernel.
- **~1.6x Amdahl arithmetic:** correct. MUL_MAT = dequant 23.7% + DNNL gemm 32.6% = 56% of prefill;
  the ~44% delta-net/attention/norms is untouched, so even a free MUL_MAT caps at ~2.3x and the
  realistic int8 case is ~1.6x. Framing is right.

## The lever that breaks the plateau (specific, implementable)

**fp16-fused GEMM: dequant Q2_0 -> fp16 weights in-register, keep activations fp16, no int
accumulation, no per-sub-block scaling.**

Mechanism - why it dodges the wall:
- Expand each Q2_0 weight tile to fp16 `(raw-1)*d_w` in-register/SLM, VNNI-packed for fp16 DPAS.
  Both scales are baked into the fp16 weight value; activations stay fp16 (no Q8_1 quantization).
- The DPAS is then a clean fp16 x fp16 -> f32 accumulate. **There is nothing between consecutive
  mads** - the float accumulator sums across all K uninterrupted, exactly like a normal GEMM. The
  per-32-K int32->float scale step that stalls the int8 kernel simply does not exist.
- Store the f32 accumulator once at the end.

Why it should reach a high peak fraction where int8 could not: it is the standard weight-only
quantized GEMM pattern (Marlin / bitsandbytes / oneDNN weights-decompression on CUDA). Those reach
70-90% of fp16 peak precisely because the decompress is amortized and the mad chain is unbroken.

Ceiling: it gives up int8's 2x (so GEMM stays ~1x fp16), but it still deletes the dequant kernel
(23.7%) and the 54 GB F16 global write. Net ceiling ~1/(1-0.237) = **~1.3x prefill**. Lower than
1.6x, but 1.3x is real and, unlike 1.6x, plausibly reachable.

Two cheaper checks before writing that kernel:
1. **Does oneDNN already do weights-decompression?** Recent oneDNN matmul supports int8/int4 weight
   decompression (quantized B, fp16 compute, internal dequant). It will not know the custom 2-bit
   Q2_0 format, but if the weights can be presented as a oneDNN-supported low-bit type, you may get
   the fused benefit with zero kernel code. Check `dnnl::matmul` weights-decompression + the
   `weights_scales`/`weights_zero_points` attrs against what DNN version ships here.
2. If not, the fp16-fused kernel above is the fallback, and it is a much better ROI than grinding
   the int8 kernel past 11 TOPS.

## Strategic consequence of the measurement (updates the recommendation)

oneDNN's GEMM is already at **68% of fp16 peak (124.7 TOPS, 0.732 ms)** - it is not leaving much on
the table on the matmul itself. The current path's only real slack is the *separate* dequant pass
plus the 54 GB F16 write. This reframes the two options:

- **fp16-fused** captures that slack ONLY IF the hand-rolled kernel matches oneDNN's ~124 TOPS GEMM
  efficiency while also dequantizing in-register. That is a high bar - oneDNN is heavily tuned, and
  hand-rolled GEMMs rarely reach 68% of peak on the first attempt. Reward ~1.3x, but the risk of
  landing below oneDNN(GEMM)+dequant is real.
- **Just speed up the standalone dequant kernel** - SUPERSEDED, see the completed-baseline section
  below. Measurement showed the dequant is already bandwidth-bound at the ceiling (~528 GB/s) after
  item 2's fix, so there is nothing left to optimize there. The 24% can only be recovered by
  ELIMINATING the dequant (fusion), not speeding it up.

Current-path total for this GEMM = 0.732 ms (oneDNN GEMM) + 0.382 ms (SoA dequant, MEASURED) =
**~1.11 ms**. fp16-fused must beat 1.11 ms *including* in-register dequant while hitting
oneDNN-class GEMM efficiency.

**Recommendation (final, measured): fp16-fused is the ONLY remaining lever for ~1.3x prefill;
standalone-dequant optimization is exhausted (bandwidth ceiling); int8-fused is dead (11x). See the
completed-baseline section for the full numbers.**

## Also worth stating plainly

The int8 approach is not salvageable to competitive by tiling tweaks. Bigger N-tiles (multiple tB)
add ILP but do not remove the per-32-K scale dependency; they would raise 11 TOPS somewhat but not
to ~150. Do not spend more on int8-fused. The decision point is fp16-fused vs "just make the
standalone dequant faster" (item 2 left ~16% on the SoA dequant). Both target the same 23.7%; the
fp16-fused kernel also removes the 54 GB write, so it has the higher ceiling of the two.

## Baseline COMPLETE 2026-07-16 (both halves measured)

For one ffn_up GEMM [K=5120, N=17408, M=512] on B70:

| component | time | note |
| --- | ---: | --- |
| oneDNN f16 GEMM | 0.732 ms | 124.7 TOPS, 68% of fp16 peak (`onednn_f16_gemm_bench.cpp`) |
| Q2_0->f16 dequant (AoS) | 0.411 ms | 491 GB/s, bandwidth-bound (`dequant_bench.cpp`) |
| Q2_0->f16 dequant (SoA, deployed) | 0.382 ms | 528 GB/s, bandwidth-bound |
| **current-path total** | **~1.11 ms** | dequant + GEMM |
| fused int8 kernel | 8.0 ms | 11x the GEMM alone - DEAD |

**Critical finding: the dequant is bandwidth-bound at the memory ceiling (~500-528 GB/s),
dominated by the 178 MB F16 output write.** After item 2's coalescing fix it is essentially optimal
as a standalone kernel - you cannot speed up a kernel already at the bandwidth ceiling.

### This CORRECTS the previous recommendation

The earlier "just make the standalone dequant faster is better ROI" was WRONG - dequant is already
at the ceiling, nothing left to get. The ~24% dequant cost can only be recovered by ELIMINATING it,
i.e. fusion. So:

- **item 2 (standalone dequant optimization) is EXHAUSTED** - it is bandwidth-bound and done.
- **fp16-fused is the ONLY remaining lever for the ~1.3x** - dequant weights in-register (no 178 MB
  write, no separate 0.4 ms pass), GEMM at fp16. It must match oneDNN's 124.7 TOPS GEMM efficiency
  while folding in the dequant. If it lands at oneDNN-class GEMM speed, it saves the full 0.4 ms
  dequant -> ~1.3x prefill. If it lands below oneDNN's GEMM, it can still win as long as
  (fused time) < 1.11 ms (GEMM+dequant). That 1.11 ms is the hard bar.
- int8-fused: dead (measured 11x).

Note the fused kernel does NOT speed up the GEMM itself (compute-bound at 124 TOPS); it only
removes the separate dequant pass and the F16 round-trip. So the entire prize is the 0.4 ms
dequant, i.e. ~24% of the MUL_MAT path -> ~1.3x prefill ceiling, confirmed by measurement.

### Original oneDNN test design (executed):

Goal: measure the current path's real cost for one ffn_up GEMM = **oneDNN F16 GEMM time + Q2_0->F16
dequant time**, and compare to the fused kernel's 8 ms. The fused kernel replaces both, so both must
be in the baseline.

Standalone oneDNN matmul benchmark:
- Dims: M=512 (tokens), K=5120, N=17408. a = activations [M,K] f16, b = weights [K,N] f16,
  c = [M,N] f32. (Match ggml's orientation: it computes dst[N,M] = W[N,K].X[K,M]; either transpose
  is fine as long as dims/FLOP match - 2*M*N*K = 91 GFLOP.)
- Use `dnnl::matmul` with f16 a/b, f32 accumulation, engine = the GPU (level_zero:1), the SAME
  oneDNN the SYCL backend links (`libdnnl.so.3` in the image / build/bin). Reuse the SYCL queue via
  `dnnl::sycl_interop::make_engine/stream`.
- Warm up 3 iters, time 20, report ms/iter and TOPS (91e9 / (ms/1e3) / 1e12).
- Separately time the dequant: reuse the existing `dequantize_row_q2_0_sycl` (or the reorder
  variant) on a [K,N] Q2_0 -> f16 buffer, same warmup/timing. Or read it from the profile
  (dequant = 23.7% of prefill) as a cross-check.
- Report three numbers: oneDNN GEMM ms, dequant ms, sum. Compare sum to the fused 8 ms.

Expected outcome and what it means:
- If oneDNN GEMM alone is ~1-1.5 ms: confirms int8-fused (8 ms) loses badly -> NO-GO stands.
- If (GEMM + dequant) is, say, ~2-3 ms: that is the bar the **fp16-fused** kernel must beat. It
  should, since it does the GEMM once at fp16 rate with no separate dequant pass and no 54 GB write
  - target < the oneDNN GEMM time plus a small dequant-in-register overhead.
- Also capture oneDNN's achieved TOPS: it calibrates what "a good fp16 kernel" looks like on this
  GPU, which is the realistic target for the fp16-fused kernel (not the 183 peak).

Do NOT re-benchmark the int8-fused kernel or re-run anything on the "already tried" list; this test
is only the oneDNN F16 baseline + dequant, which has never been isolated.
