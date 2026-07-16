# Ternary-Bonsai-27B on Intel Arc (SYCL) - working notes

Running log for this fork's SYCL work: what we are doing, what we tried, what it
measured, and what is next. Append to it; do not rewrite history.

Fork: `meatposes/llama.cpp` (fork of `PrismML-Eng/llama.cpp`, itself a fork of `ggml-org/llama.cpp`).
Work branch: `sycl/bonsai-q2_0-perf`.
Targets: Arc Pro B70 (`bmg_g31`, 32656 MiB), Arc Pro B50 (`bmg_g21`, 16 GiB).
Model: `prism-ml/Ternary-Bonsai-27B-gguf`, `Ternary-Bonsai-27B-Q2_0.gguf`.

---

## 1. The model - read this before optimizing anything

Getting this wrong cost the first round of work its direction, so it goes first.

**Bonsai is not a dense attention model.** `qwen35` is a hybrid architecture. Confirmed from
GGUF metadata, the tensor list, and `src/llama-model.cpp`; the prism-ml model card agrees
("~75% linear attention and 25% full attention across 64 blocks").

| Property | Value | Source |
| --- | --- | --- |
| `qwen35.full_attention_interval` | 4 | GGUF metadata |
| `qwen35.block_count` | 64 | GGUF metadata |
| Full-attention layers | **16** (blk.3, 7, 11, ... 63) | interval = 4 |
| Gated-delta-net layers | **48** (blk.0, 1, 2, 4, ...) | remainder |
| Hybrid dispatch | `llama_memory_hybrid` + `is_recr(il)` filter | `src/llama-model.cpp:2075` |
| SSM params | `state_size=128`, `inner_size=6144`, `group_count=16`, `conv_kernel=4` | GGUF metadata |
| Attention | `head_count=24`, `head_count_kv=4`, `key_length=value_length=256` | GGUF metadata |
| RoPE | IMROPE, `dimension_sections=[11,11,10,0]`, `freq_base=1e7` | `src/llama-model.cpp:2496` |
| Vocab / output head | `output.weight` Q2_0 `[5120 x 248320]`, ~318 MB | GGUF tensors |
| MTP head | **none** - do not chase this | model card + tensor list |

Check it yourself: `blk.0` carries `ssm_a`, `ssm_alpha/beta.weight`, `ssm_conv1d.weight`,
`ssm_dt.bias`, `ssm_norm.weight`, `ssm_out.weight`. `blk.3` carries `attn_q/k/v/output` and
`attn_q_norm/k_norm`. That is the hybrid split, visible directly in the tensor names.

**Why it matters:** every optimization so far (Q2_0 MMVQ, SoA reorder, the abandoned MMQ,
oneDNN F16, AOT) targets `MUL_MAT`. Nothing has touched the `GGML_OP_GATED_DELTA_NET` path,
which runs in **48 of 64 layers**. Profile before assuming `mmq.cpp` is where the time goes.

### Q2_0 format

Ternary `{-1, 0, +1}`, ~1.71 bits/weight. 128 elements per block (`QK2_0=128`), 2 bits each
(4 per byte), one FP16 scale per block. Stored raw values are `{0,1,2,3}`; the true weight is
`raw - 1`. The vec_dot applies that bias with the Q8_1 row sum (`ds8f.y()`) rather than
materializing shifted values:

    sum((raw-1) * q8) = sum(raw * q8) - sum(q8)

### KV cache math

Per token, per full-attention layer: K = `4 kv_heads * 256 dim * 2 B` = 2048 B, V likewise.
4096 B/layer, times **16** attention layers = **64 KiB/token**.

- `-c 131072` -> **8 GiB** of KV, on top of 6.7 GiB of weights. Fits a B70; `-fit on` is
  silently adjusting.
- **Does not fit a B50 (16 GiB).** KV quantization is mandatory there, not optional.

---

## 2. Current state

Committed on `sycl/bonsai-q2_0-perf`:

| Commit | What |
| --- | --- |
| `sycl: add Q2_0 support` | dequant, get_rows, MMVQ, SoA reorder, dead MMQ case |
| `sycl: drop blocking waits from contiguous concat` | graph-recording fix |
| `sycl: raise MMVQ_MAX_BATCH_SIZE from 8 to 32` | validated 2026-07-16: ~2x at batch 16-32 |
| `sycl: make the MMVQ batch cap runtime-tunable` | `GGML_SYCL_MMVQ_MAX_BATCH` |
| `sycl: embed SPIR-V fallback alongside AOT` | `spir64_gen,spir64` + multi-device |

### Build

    source /opt/intel/oneapi/setvars.sh --force
    cmake -S . -B build \
          -DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=ON \
          -DGGML_SYCL_DEVICE_ARCH="bmg_g31,bmg_g21"
    cmake --build build --config Release \
          --target ggml-sycl llama-server llama-bench -j$(nproc)

AOT takes 20-60 min per device target. Output `.so` was ~193 MB for a single AOT target;
expect roughly 2x for two, plus the spir64 fallback.

`libdnnl.so.3` must be copied into the image at `/app/` (it is on `LD_LIBRARY_PATH`).
Do **not** work around a missing DNNL by setting `GGML_SYCL_DNN=OFF` - that silently disables
the F16 path, which is the single largest prefill win.

### Runtime knobs

`ggml/src/ggml-sycl/ggml-sycl.cpp:257-275`, with in-code defaults:

| Variable | Default | Note |
| --- | --- | --- |
| `GGML_SYCL_MMVQ_MAX_BATCH` | 32 | ours; clamped to [1,32]; upstream constant was 8 |
| `GGML_SYCL_DISABLE_GRAPH` | 1 | prism default is *disabled*; set 0 to enable |
| `GGML_SYCL_ENABLE_FLASH_ATTN` | 1 | |
| `GGML_SYCL_DISABLE_DNN` | 0 | |
| `GGML_SYCL_PRIORITIZE_DMMV` | 0 | |
| `GGML_SYCL_USE_ASYNC_MEM_OP` | 1 | |
| `GGML_SYCL_ENABLE_VMM` | 1 | |
| `GGML_SYCL_DISABLE_OPT` | 0 | disables the SoA reorder |

---

## 3. What we tried

### Worked

**Q2_0 SoA reorder (root TG win).** Two bugs kept it from ever running: Q2_0 was missing from
the `init_tensor` extras switch, so no `ggml_tensor_extra_gpu` existed and `opt_for_reorder()`
bailed at its `if (!extra || ...)` guard; and Q2_0 was missing from the `reorder_qw()` switch,
which would have hit `GGML_ABORT` if reached. Both fixed.

**`GGML_SYCL_F16=ON` + `GGML_SYCL_DNN=ON`.** Routes prefill GEMM through oneDNN with
`fpmath_mode::f16` onto the Xe2 XMX engines. Largest prefill win. Requires DNNL present.

**`GGML_SYCL_DEVICE_ARCH=bmg_g31`.** AOT; 90s -> 5.7s cold start. The CMake block for this is
native to the prism fork, not something we added.

**concat without blocking waits.** `concat_impl_sycl()` called `.wait()` after each memcpy in
the contiguous path, which blocks SYCL graph recording. The queue is created with
`sycl::property::queue::in_order` (`dpct/helper.hpp:748, 787`), so ordering was already
guaranteed and the waits were redundant. Not a hack; upstreamable.

### Did not work

**Custom Q2_0 MMQ kernel.** Reverted. dp4a MMQ lost badly to dequant + oneDNN F16 GEMM:

| Config | pp512 |
| --- | --- |
| dequant + oneDNN F16 (kept) | **861 t/s** |
| MMQ, 64x64x4 tiles | 130 t/s |
| MMQ, 16x64x4 tiles | 153 t/s |

Caveat on this verdict: our `VDR_Q2_0_Q8_1_MMQ` was **1**, while CUDA's is **4**
(`ggml/src/ggml-cuda/vecdotq.cuh:113`) - a quarter of the work per dp4a iteration, so 4x the
loop overhead. The conclusion is probably still right (dp4a cannot beat XMX), but the evidence
is weaker than it looks. `mmq.cpp` still has a Q2_0 case; it is dead code without
`GGML_SYCL_FORCE_MMQ`.

**KV cache quantization (`-ctk q8_0 -ctv q8_0`).** Measured 2026-07-16 on B70, `-fa 1`, `-r 2`.
A large regression that worsens with depth. **Keep F16 KV.**

| depth | f16 KV | q8_0 KV | delta |
| ---: | ---: | ---: | ---: |
| 0 | 42.54 +/- 0.24 | 40.33 +/- 0.18 | -5.2% |
| 4096 | 39.02 +/- 0.05 | 31.56 +/- 1.87 | -19.1% |
| 16384 | 32.27 +/- 0.04 | 21.90 +/- 0.02 | -32.1% |
| 32768 | 25.97 +/- 0.01 | **15.21 +/- 0.01** | **-41.4%** |

This is not a bandwidth effect: q8_0 reads *half* the KV bytes, so a bandwidth-bound attention
would get faster. It is the FA kernel selection (`ggml_sycl_get_best_fattn_kernel`,
`fattn.cpp:195-208`):

- **F16 KV:** VEC is taken only `if (!gqa_opt_applies)`. With `gqa_ratio = 24/4 = 6`, a mask and
  `max_bias == 0`, `gqa_opt_applies` is true, so TG falls through to **TILE** - which exploits GQA
  and shares each KV read across 6 Q heads.
- **Quantized KV:** returns **VEC** unconditionally at `Q->ne[1] <= 2`, ignoring
  `gqa_opt_applies`, losing the GQA sharing. ~6x more KV work swamps the 2x saved by quantizing.

**This is not a portable one-line fix.** The TILE kernel is F16-only - neither
`ggml/src/ggml-sycl/fattn-tile.cpp` nor `ggml/src/ggml-cuda/fattn-tile.cu` contains any quantized
type handling. Quantized KV *must* go to VEC. The asymmetry in the selection code is a
consequence of that, not an oversight. Note CUDA does guard its F16 VEC path against this regime
(`!(gqa_ratio > 4 && K->ne[1] >= 8192)`, `fattn.cu:460`), i.e. upstream knows VEC is bad for
high-GQA long-context.

Consequences:

- **B50 (16 GiB): do not reach for KV quant to fit.** Prefer F16 KV at reduced context. F16 at
  `-c 65536` is 4 GiB (6.7 + 4 + buffers, comfortable). F16 at `-c 131072` is 8 GiB (6.7 + 8 +
  buffers ~= 15.7 GiB, marginal-to-OOM). q8_0 at 131072 fits but costs ~41% TG at depth.
- Making quantized KV viable needs a GQA-aware quantized TILE (or XMX) kernel, not a dispatch
  tweak. Same bucket as the `fattn.cpp:192` XMX TODO.

**dspark speculative decoding.** Measured 84.9% acceptance but TG 41 -> 17 t/s (-59%).
**Do not trust the recorded root cause** - see open questions.

**SYCL graph.** Neutral. An earlier "harmful" result was a concurrent-instance artifact.

### Measured (llama-bench, B70, single instance, graph off, AOT bmg_g31, F16, SoA)

| Test | t/s |
| --- | --- |
| pp128 / pp256 / pp512 | 419 / 671 / **861** |
| pp1024 / pp2048 | 847 / 813 |
| tg128 @ d=0 / d=3072 | 42.1 / 39.6 |
| pp512 @ d=0 / d=3072 | 861 / 528 |

Cold start 5.7s. First run includes the one-time reorder (~563 t/s pp512); steady state is 861.

**These numbers do not describe the deployed server.** llama-bench runs one sequence, default
context, graph off. The server runs `-c 131072`, `n_parallel=4` (auto), graph on, prompt cache,
and context checkpoints - and logs **25.6 t/s** eval. Always state which harness produced a number.

---

## 4. Open questions

### The dspark contradiction (highest value)

The prism-ml model card reports dspark at **1.34x decode on CUDA** (H100: 98 -> 131.8 t/s),
lossless, via a 6-layer semi-autoregressive drafter tapping hidden states from 5 target layers
(`target_layers=[1,16,31,46,61]`, `block_size=4`). We measured **-60% on SYCL**. Opposite sign.

The recorded verdict ("27B draft model overhead exceeds acceptance benefit") is almost certainly
wrong as a root cause - a drafter winning 1.34x on CUDA at 84.9% acceptance does not lose 60% on
Arc for architectural reasons.

Prime suspect: dspark uses context checkpoints, and on a hybrid model each draft round may
snapshot and/or roll back **~150 MiB of recurrent state**. `create_checkpoint()`
(`tools/server/server-context.cpp:2187`) calls `update_tgt(...)` -> `llama_state_seq_get_data`
into a **host** buffer, i.e. device-to-host over PCIe. On an H100 that copy is trivial; on B70
over PCIe it is ~15-30 ms, against a ~24 ms/token budget. The magnitude matches the loss.

Note the model card says the fork officially supports **CUDA and Metal only**. SYCL is ours.

### Is MMVQ at batch 9-32 actually faster than oneDNN? - ANSWERED: yes, ~2x

**Measured 2026-07-16. Keep the cap at 32; do not revert `342202a46`.**

Setup: B70 (`level_zero:1`), JIT build, F16 + DNNL on, FORCE_MMQ off, `-b 512 -p 512 -n 0 -r 3`,
sweeping `-ub`. `ub=8` is the control: it is <= both caps, so it must land in the MMVQ path under
each and come out identical. It did (-0.6%), which validates the design.

| n_ubatch | cap=8 (t/s) | cap=32 (t/s) | speedup | path under cap=8 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 170.03 +/- 0.32 | 168.95 +/- 0.07 | 1.00x (control) | MMVQ (both) |
| 16 | 84.12 +/- 0.24 | **189.96 +/- 0.10** | **2.26x** | oneDNN |
| 24 | 93.41 +/- 0.01 | **168.08 +/- 0.04** | **1.80x** | oneDNN |
| 32 | 98.24 +/- 0.02 | **204.21 +/- 0.09** | **2.08x** | oneDNN |

The prior suspicion that 32 was a regression came from an old `pp32 = 120 t/s` data point that
predates the SoA reorder and F16. It was wrong.

**Mechanism:** dequant+oneDNN must dequantize the *entire* weight matrix regardless of batch
size. At ubatch 16 that means dequantizing ~6.7 GB of weights to serve a 16-column GEMM - fixed
cost, negligible work. MMVQ reads the quantized weights directly and skips it. The same effect
explains pp512 = 861 t/s: at ubatch 512 the dequant cost finally amortizes.

**Scope:** the cap is irrelevant to normal prefill (ubatch 512 -> oneDNN wins by a wide margin).
It matters for small-batch decode: `n_parallel` serving (batch = active slots) and speculative
decoding (batch ~= draft block size 4-5). Both sit in the 2x band.

Follow-ups this opens:

- **The crossover is well above 32.** MMVQ@32 = 204 vs oneDNN@32 = 98, and oneDNN does not reach
  861 until ubatch 512. 32 is not an optimum - it is just where `*_switch_ncols` stops
  instantiating. Extending to 64 (and raising `MMVQ_MAX_BATCH_SIZE_LIMIT`) may gain more, at the
  cost of more kernel instantiations, build time and `.so` size. Worth bisecting where MMVQ and
  oneDNN actually cross.
- **ub=24 (168) is below ub=16 (190) and ub=32 (204).** Non-monotonic; likely a tail/occupancy
  effect on a non-power-of-2 ncols. Minor, but it means the MMVQ ncols kernels are not uniformly
  tuned.

Caveat: `llama-cpp-sycl` (spans `level_zero:0;1`) was resident but idle during these runs, so it
held VRAM on the device. It affected both arms equally.

### Where does the time actually go?

Unknown, and it gates everything else. The delta-net path (48/64 layers) has never been
profiled. Prior VTune runs predate the F16/reorder work and only ever showed
`sgemm_nocopy_tn_32x16_4x8`, i.e. the old dequant+BLAS path.

- Host VTune: `/opt/intel/oneapi/vtune/latest/bin64/vtune`
- GPU PMU needs `sudo sysctl dev.i915.perf_stream_paranoid=0`
- Without the kernel module, XVE utilization reads 0% but kernel timing is still accurate

Roofline: TG 42 t/s x 6.7 GiB = ~281 GB/s achieved. Confirm the B70's spec bandwidth. If peak is
~450 GB/s we are at ~62% and TG headroom is real; if the delta-net scans dominate, TG is
latency-bound and no amount of `MUL_MAT` work will help.

### Context checkpoints

~150 MiB per checkpoint (the recurrent state of 48 delta-net layers: 16 groups x 128 state x 384
head-dim x 4 B x 48 layers ~= 151 MiB, matching the observed 149.626 MiB), at `min spacing = 256`
tokens. On a 4096-token prompt that is ~16 device-to-host copies, ~2.4 GiB of PCIe traffic during
prefill. llama-bench does none of this.

**Do not simply disable them.** Recurrent state cannot be rewound to an arbitrary position, so on
a prompt-cache miss the only alternative to restoring a checkpoint is reprocessing the whole
prompt. They are load-bearing for multi-turn TTFT. Tunables: `-ctxcp N`
(`LLAMA_ARG_CTX_CHECKPOINTS`, default 32) and `-cms N` (`--checkpoint-min-step`, default 256).
Sweep against a realistic multi-turn workload and measure both throughput and TTFT.

---

## 5. Next steps

P0 - free, no rebuild:

1. ~~`-ctk q8_0 -ctv q8_0`.~~ DONE - **rejected**, -41% TG at d=32768. Keep F16 KV. See section 3.
2. Sweep `-cms` / `-ctxcp` against a multi-turn workload.
3. Build a benchmark that mirrors the deployment (`-c`, `n_parallel`, graph, prompt cache) and
   re-baseline. The 861/42.1 numbers are not what production does.

P1 - cheap, resolves open questions:

4. ~~A/B `GGML_SYCL_MMVQ_MAX_BATCH` 8 vs 32.~~ DONE - 32 wins ~2x at batch 16-32. Next: find the
   real MMVQ/oneDNN crossover above 32 (needs `*_switch_ncols` instantiated past 32).
5. Sweep `-ub` / `-b`. `n_ubatch=512` is an untouched default and 48 sequential-scan layers make
   512 non-obvious.
6. Get a B50 baseline on the dual-arch build.
7. Establish the bandwidth ceiling. **This decides whether P2/P3 are worth doing at all.**

P2 - profile:

8. VTune the current build. Per-kernel split of TG and PP across delta-net vs FA vs MUL_MAT.
9. Check the output head: Q2_0 `[5120 x 248320]`, ~318 MB, a 248320-row GEMV every token.

P3 - real engineering, only if P2 justifies it:

10. Chunked gated-delta-net kernel for prefill. Both backends scan tokens sequentially
    (`ggml/src/ggml-sycl/gated_delta_net.cpp:70`, `ggml/src/ggml-cuda/gated_delta_net.cu:63`),
    and the CUDA file carries the TODO explicitly:
    `//TODO: Add chunked kernel for even faster pre-fill` (`gated_delta_net.cu:183`).
    So there is **nothing better to port from CUDA** - SYCL is at parity. Highest ceiling in this
    document, backend-agnostic; doing it in CUDA first would be upstreamable and easier to validate.
11. XMX flash attention. `ggml/src/ggml-sycl/fattn.cpp:192` says `// Todo: Use the XMX kernel if
    possible:` - attention currently runs on generic SIMD tile/vec kernels while GEMMs get XMX via
    oneDNN. With `head_dim=256` this is expensive and likely explains the 861 -> 528 t/s falloff
    with depth. Large job.
12. Do **not** restart MMQ unless P2 shows `MUL_MAT` is the bottleneck *and* you intend to use
    `joint_matrix`/XMX rather than dp4a. If you do, fix `VDR_Q2_0_Q8_1_MMQ` to 4 first.

---

## 6. Flash attention notes

FA defaults to on (`flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO`, `common/common.h:487`), so it
is already in every number above.

Kernel selection for this model (`ggml_sycl_get_best_fattn_kernel`, `fattn.cpp:104`):

- `head_dim = 256` is supported (it is in the `K->ne[0]` switch).
- `gqa_ratio = 24/4 = 6`, so with F16 KV, a mask and `max_bias == 0`, `gqa_opt_applies` is true
  and TG (`Q->ne[1] == 1`) falls through to **TILE**.
- With quantized KV, the `Q->ne[1] <= 2` branch returns **VEC** instead.

So **KV quantization silently changes which FA kernel you run.** Benchmark both; do not assume.

`GGML_SYCL_FA_ALL_QUANTS` has **no CMake option** in this fork (unlike `GGML_CUDA_FA_ALL_QUANTS`
at `ggml/CMakeLists.txt:210`), so only matching K/V pairs work: F16/F16, Q4_0/Q4_0, Q8_0/Q8_0.
Mixed types (e.g. q8_0/f16) need that option added first.

---

## 7. Intel A-series (Alchemist / ACM) support

**Status: should work via the SPIR-V JIT fallback; unverified - no A-series card in this machine**
(the box has 2x B70 plus an Arrow Lake iGPU on `level_zero:2`).

The good news is that A-series needs **no new backend code**:

- `gpu_arch` is just SYCL's standard `sycl::ext::oneapi::experimental::architecture` enum, and
  `acm_g10` / `acm_g11` / `acm_g12` are **already** in the detection map
  (`ggml/src/ggml-sycl/sycl_hw.cpp:26-28`), classified `GPU_FAMILY_DGPU_CLIENT_GAME`.
- Alchemist (Xe-HPG) **has XMX**, so the oneDNN F16 prefill path should engage as it does on
  Battlemage.
- It has dp4a, so MMVQ works.
- `GGML_SYCL_WARP_SIZE=16` is set for all `INTEL` targets; ACM supports sub-group size 16.

What to do to actually get it running:

1. **Nothing, if the SPIR-V fallback is enough.** Since commit `d699fd213` the AOT build embeds
   generic SPIR-V, so an A-series card JIT-compiles at startup (~90s cold start) and runs.
   This is why we did not pay AOT cost for hardware we cannot test.
2. **For AOT (fast cold start), add the device:**

       -DGGML_SYCL_DEVICE_ARCH="bmg_g31,bmg_g21,acm_g10"

   Each AOT device recompiles the whole kernel set: expect roughly +1x build time and +190 MB
   `.so` per target. Only worth it on a machine that actually has the card.

Things to check when an A-series card is available:

- **VRAM.** A770 is 16 GiB, same constraint as B50: `-c 131072` needs 8 GiB of KV and will not
  fit alongside 6.7 GiB of weights. Use `-ctk q8_0 -ctv q8_0`, and expect to lower `-c`.
- **There is an existing acm_g10 special case in dispatch**
  (`ggml/src/ggml-sycl/ggml-sycl.cpp`, `ggml_sycl_mul_mat`): "Arc770 get benefit with Q4_0 by
  skipping it" skips the MMVQ-over-DMMV preference for `acm_g10` + Q4_0. It does not apply to
  Q2_0, but it is evidence that ACM has its own dispatch quirks - re-tune
  `GGML_SYCL_MMVQ_MAX_BATCH` there rather than assuming Battlemage's answer carries over.
- **Xe-HPG XMX is narrower than Xe2's.** The F16 GEMM will engage but the B70 tuning (tile sizes,
  the MMVQ/GEMM crossover) should not be assumed to transfer.
- Alchemist needs `-ze-intel-greater-than-4GB-buffer-required` for >4 GiB allocations; note the
  CMake currently skips that flag whenever `GGML_SYCL_DEVICE_ARCH` is set
  (`ggml/src/ggml-sycl/CMakeLists.txt:167-171`). It has not bitten us on B70 because weights are
  allocated per-tensor rather than as one >4 GiB buffer, but confirm on a 16 GiB card.
