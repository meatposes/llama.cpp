# SoA dequant layout change - plan + running log (started 2026-07-17)

## Goal
Close the ~16% server-prefill gap: post-reorder (SoA) dequant runs at ~769 t/s vs the AoS ceiling
~916. Root cause (proven earlier, section 3b of the main notes): the SoA layout splits reads into
two distant streams - `qs` at offset 0, scales `d` at offset `k/4` - while AoS keeps `d`+`qs`
together (one stream). Fix = a layout that gives the dequant one-stream locality WITHOUT regressing
MMVQ (which needs `qs` contiguous across blocks and is worth 2.7x TG).

## The tension (why it is not trivial)
- **AoS** `block_q2_0` = `[d(2B) qs(32B)]` per block: one stream for dequant (fast, 916), but MMVQ
  reads `qs` strided by 34B across blocks -> 2.7x slower TG. Rejected.
- **SoA** (current) = `[qs0 qs1 ... qsN][d0 d1 ... dN]`: MMVQ reads `qs` contiguous (fast TG), but
  dequant reads two distant regions (769).
- Need a layout that is ~one-stream for dequant AND keeps `qs` ~contiguous for MMVQ.

## Candidate layout: superblock-interleaved
`[qs of S blocks (S*32B)][d of S blocks (S*2B)]` repeating. MMVQ reads S contiguous qs blocks
(S*32B runs) then skips S*2B of d - mostly contiguous. Dequant gets d LOCAL to its qs (within
S*34B), not k/4 away. Sweep S in {1,2,4,8,16,32}. S=nblocks = current SoA; S=1 ~= AoS. The optimum
trades MMVQ-contiguity vs dequant-locality.

## Phases
- **P0** Standalone baseline: generate a real ffn_up-size Q2_0 weight; measure dequant (AoS vs
  current-SoA) and MMVQ (current-SoA) throughput + correctness. Confirm the gap standalone.
- **P1** Prototype superblock layout in the standalone bench; sweep S; measure BOTH dequant AND
  MMVQ per S. Winner must improve dequant with <=~1% MMVQ regression. If none wins both -> document
  the tradeoff as fundamental and stop (valid outcome).
- **P2** If a winner: implement in the real backend - `reorder_qw_q2_0` (AoS->layout),
  `dequantize_block_q2_0_reorder`, `reorder_vec_dot_q_sycl<Q2_0>`, `block_q_t<Q2_0>` offsets
  (quants.hpp). Build. Correctness: `test-backend-ops -o {GET_ROWS,MUL_MAT} -b SYCL0` + generate
  through post-reorder prefill. A/B full model: pp512 server-path (`-ub 8,512`) AND tg128.
- **P3** Only if pp up AND tg flat: build image, validate (health, cold start, correctness), and
  leave a DEPLOY RECOMMENDATION for the user (do not silently swap the live server unless clearly
  net-positive with zero TG regression; keep `:meat4-dspark` as rollback).

## Safety rules for the autonomous run
- Live `llama-cpp-bonsai:meat4-dspark` stays deployed until a validated win. Rollback image retained.
- Stop bonsai only for a bench, with a trap to restart it; never leave it down.
- One `DEVICE_LOST` = stop that line of work (GPU-hygiene rule).
- Commit + push after each phase; keep this log and the main notes current.
- Correctness gate before any perf claim; A/B same-binary where possible.

## Running log
(appended below as phases complete)

## P0 RESULT (2026-07-17) - premise challenged, re-verifying

Standalone layout sweep (`soa_layout.cpp`, ffn_up size): **dequant is FLAT at 0.713 ms across ALL
S** (S=1 AoS through S=nblocks current-SoA). The layout does NOT change dequant kernel speed in
isolation. MMVQ-read: S=1 (AoS) slow (149 GB/s, strided), S>=2 all fast (~420 GB/s).

This contradicts the "two-stream read costs the dequant 16%" premise. It also matches the item-2
review where standalone SoA dequant (0.382 ms) was already ~= AoS (0.411 ms). So the full-model
916 (reorder-off) vs 769 (reorder-on) prefill gap is probably NOT the dequant layout.

**Pivot: re-profile the reorder-ON prefill to find the REAL source of the gap before building any
layout change.** If the dequant kernel is not the difference, a layout change won't help and this
task becomes "find what actually causes the reorder-on prefill penalty."

## P1 RESULT (2026-07-17) - NOT a layout problem, a WRITE-VECTORIZATION problem. FIX FOUND.

Profiled the reorder-ON prefill: SoA dequant = 0.494 s vs AoS generic-template dequant = 0.268 s
(same DNNL gemm ~0.37 s both). So the gap IS the dequant kernel, but not the layout - the AoS
generic template uses vectorized (dfloat2) writes while my SoA kernel used 4 scalar half stores.

Standalone kernel comparison on the SAME SoA layout (`dq_kernel.cpp`):
  K1 (4 scalar writes, current) : 0.801 ms, 223 GB/s
  K2 (2x half2 write)           : **0.374 ms, 477 GB/s  (2.1x)**
  K3 (2 bytes, 4x half2)        : 0.715 ms (worse - occupancy)

**Fix: write the 4 dequant outputs as two `sycl::vec<dst_t,2>` instead of 4 scalars.** No layout
change, no MMVQ touch, no TG risk. Applied to `dequantize_block_q2_0_reorder`. Expected: SoA dequant
0.494 -> ~0.27 s, closing most of the 769 -> 916 server-prefill gap (~+16%).

## P2 RESULT (2026-07-17) - VALIDATED, big win, deploying

Full-model A/B (JIT build, fixed kernel):
- **server-path pp512: 768 -> 994 (+29%)** - beats even the old AoS-path 916 (the half2 SoA dequant
  is faster than the AoS generic template).
- **tg128: 42.22 (unchanged)** - zero TG regression, as expected (dequant is the prefill path).
- Correctness: coherent generation including through the post-reorder prefill path ("Paris", a
  50x-repeated long prompt answers "fox", "1..10"). Standalone K2 also matched the CPU reference.

The task turned out NOT to be the layout change I planned - it was a one-line write-vectorization
in the existing SoA dequant kernel. Lower risk, bigger win. Layout change abandoned (unneeded).
Next: AOT rebuild + image + deploy (keeping :meat4-dspark as rollback).
