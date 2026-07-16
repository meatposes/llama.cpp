# Review request: XMX Q2_0 GEMM NO-GO decision

**For a reviewing model.** Scope is ONLY the fused int8 XMX Q2_0 GEMM investigation and
its NO-GO verdict. Do not re-review the rest of the project. Background you may need is already in
`docs/sycl-bonsai-notes.md` (section 6 has the full write-up; sections 3b/4 have the prefill
profile) and the runnable probes in this directory - read those instead of asking for a re-explain,
and do not propose re-running experiments listed under "Already tried" below.

## The question being decided

For Ternary-Bonsai-27B (Q2_0, hybrid qwen35) prefill on Intel Arc B70 (Battlemage, bmg_g31):
can a **fused int8 XMX GEMM that consumes Q2_0 weights directly** beat the current path
(dequantize Q2_0 -> F16, then oneDNN F16 GEMM)?

The prize is real: dequant is ~24% of prefill and int8 DPAS is ~2x fp16 DPAS, so the analytic
ceiling is ~1.6x prefill (derivation in section 6). The question is purely whether a kernel can
realize it.

## What was measured (facts, not opinion)

- int8 DPAS peak on B70: **358 TOPS**; fp16 DPAS: 183 TOPS; ratio 1.95x. (`xmx_peak_int8_vs_fp16.cpp`)
- The fused kernel is fully correct at real Bonsai size (K=5120, N=17408, M=512), maxrel 0.
- Throughput across 5 engineering rounds: 1.7 -> 5.0 -> 6.5 -> **11.3 TOPS**, then plateau.
  MSTRIP (m-tiles per sub-group) > 4 regresses (3.9 / 4.8) due to register spill of the persistent
  float accumulators.
- Best kernel: `q2_tiled_inreg_scale.cpp` (MSTRIP=4, in-register `get_coord` scaling).
- 11.3 TOPS = ~3% of the 358 int8 peak. This one ffn_up GEMM: ~8 ms here vs an estimated ~1 ms for
  oneDNN F16 (assuming oneDNN hits ~50% of 183 TOPS). So the fused kernel loses the matmul ~8x even
  though it skips dequant.

## Design (so you can critique it, not reconstruct it)

- Expand Q2_0 to int8 `(raw - 1)` in {-1,0,1,2}; signed int8 DPAS needs no bias term.
- Activations are Q8_1 (int8 + per-32 scale), loaded directly from global into the A tile.
- Weights expanded to VNNI-packed int8 in SLM, once per (N-tile, 32-K sub-block), reused across the
  m-strip. B operand requires VNNI (`ext_intel_packed`).
- Tile M8 x N16 x K32 (the int8 DPAS shape). int32 accumulator.
- Scaling: per 32-K sub-block, scale the int32 result by `d_x[m] * d_w[n]` (in-register via
  `get_coord`) and add into a persistent float accumulator. This is forced per-sub-block because
  Q8_1's activation scale changes every 32 K.

## Why it plateaus (my diagnosis - challenge it)

1. **Per-32-K scale stalls the XMX pipeline.** Every DPAS is followed by an int32->float scale
   before the next accumulate. Q8_1's per-32 scaling forces this cadence.
2. **Register pressure caps the tile.** Persistent float accumulators to amortize weight expansion
   spill past MSTRIP=4, so the tile stays tiny and overhead-per-DPAS stays high.
3. Tiny M8xN16 tile means expansion + barrier + activation load overhead dominates the single DPAS.

## Where I want your review (specific)

1. **Is 11.3 TOPS actually the ceiling of this approach, or did I miss an obvious optimization?**
   Concretely: (a) can the per-sub-block scale be avoided/deferred - e.g. accumulate int32 across a
   full 128-K Q2_0 block and apply the 4 Q8_1 sub-block scales differently (Q8_1 stores a per-block
   sum; is there a decomposition that defers scaling to per-128-K)? (b) a register-tiling scheme
   that grows the tile without spilling (larger N via multiple tB, fewer persistent accumulators)?
2. **Is the NO-GO justified, or is the gap just my kernel being unoptimized?** oneDNN is heavily
   tuned; my kernel is hand-rolled. Is ~8x plausibly closable, or is the per-32-K scaling a hard
   architectural wall for int8-fused on XMX?
3. **Is there a better framing than int8-fused?** Two alternatives NOT tried:
   (a) fp16 XMX with in-register Q2_0->fp16 dequant (no per-block int scaling stall, but 1x not 2x
   throughput - does removing the stall beat the lost 2x?);
   (b) don't fuse at all - just make the standalone dequant kernel faster (item 2 got the SoA
   dequant +16% but left ~16% on the table; is attacking dequant a better ROI than the fused GEMM?).
4. **Sanity-check the ~1.6x ceiling arithmetic** (section 6): MUL_MAT = 56% of prefill
   (dequant 24% + gemm 33%), rest is delta-net/attention. Is the Amdahl framing right?

## Already tried - do NOT propose re-running these (see project notes for detail)

- dp4a MMQ (non-XMX): lost badly to oneDNN, reverted. XMX int8 is a different path.
- Raising MMVQ cap / extending switch_ncols: inert at n_parallel=4 (decode batch <=4).
- KV quantization: -41% TG, rejected.
- MMV_Y tuning for TG: 1 is optimal.
- Larger MSTRIP: regresses (register spill).
- The SoA-vs-AoS layout for the XMX feed: decided AoS (XMX prefill and MMVQ decode are separate
  dispatch paths).

## Deliverable wanted from you

A short verdict: **agree NO-GO**, or **specific approach that plausibly breaks the 11 TOPS plateau**
with enough detail to implement. If the latter, name the concrete kernel change and the expected
mechanism - not "try tuning it more." Probes here are runnable (`icpx -fsycl -fsycl-targets=
spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" -O2 <file>`, run with
`ONEAPI_DEVICE_SELECTOR=level_zero:1`).
