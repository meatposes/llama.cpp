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

**Why it matters:** it explains the shape of the model, but do **not** conclude from "48 of 64
layers" that delta-net is where the time goes. It is not - profiling (section 4) measured it at
~9.6% of prefill and ~2% of TG. They are cheap layers. Prefill time is dominated by MUL_MAT and,
above all, by **dequantizing Q2_0 weights to F16 to feed oneDNN** (23.7% of PP on its own).

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
- **B50 (16 GiB)**: arithmetic says `-c 131072` is tight (6.7 + 8 + buffers ~= 15.7 GiB), but
  **measured 2026-07-16 it starts and serves** - see the B50 section below. Do not use quantized
  KV to buy room: q8_0 costs ~41% TG at depth on this model (section 3). Prefer F16 KV at reduced
  context if you need headroom (`-c 65536` = 4 GiB).

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
| `sycl: fix work-group size in non-contiguous concat` | **+8% prefill**; generic backend fix |
| `sycl: normalize GGML_SYCL_DEVICE_ARCH for ocloc` | `_`->`-`; unblocks multi-device AOT |
| `sycl: coalesce writes in the Q2_0 SoA dequant kernel` | **+15% server prefill** (662 -> 763) |
| `sycl: one qs byte per work-item in Q2_0 SoA dequant` | +0.7% more (-> 769) |

### Build

    source /opt/intel/oneapi/setvars.sh --force
    cmake -S . -B build \
          -DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=ON \
          -DGGML_SYCL_DEVICE_ARCH="bmg_g31,bmg_g21"
    cmake --build build --config Release \
          --target ggml-sycl llama-server llama-bench -j$(nproc)

AOT takes 20-60 min per device target. `.so` is **355 MB** for two targets + spir64 fallback
(was 193 MB single-target). Both `Build succeeded for : bmg-g31.` and `bmg-g21.` should appear
in the build log.

**ocloc device-name trap (fixed in CMake, but know it exists):** ocloc acronyms are hyphenated
(`bmg-g31`). The underscore form is an accepted alias *for a single device*, which is why
`-device bmg_g31` always worked - but in a comma list ocloc drops the underscore and dies with
`Failed to parse target : bmgg31 - invalid device`. The CMake now normalizes `_` -> `-`, so
`GGML_SYCL_DEVICE_ARCH="bmg_g31,bmg_g21"` works. Verified against ocloc 26.22:

| `-device` value | result |
| --- | --- |
| `bmg_g31` | OK (single underscore alias) |
| `bmg-g31,bmg-g21` | OK |
| `bmg_g31,bmg_g21` | FAIL |
| `bmg-g31:bmg-g21` | OK (range) |

### Image build + deploy

    # .dockerignore has build*/ at line 11 - must be commented out for the COPY, then restored
    docker build -f Dockerfile.mmq-test -t llama-cpp-intel:prism-concatfix .
    docker tag llama-cpp-intel:prism-concatfix llama-cpp-bonsai:meat2

**Currently deployed: `llama-cpp-bonsai:meat3` (`f0f0e7a9ab9d`)**, 2026-07-16, includes the concat
and SoA dequant fixes. Cold start 2.93s. Rollback images retained: `:meat2` (`06c007b37469`,
concat fix only) and `:meat` (`6c00cd3a7690`, neither).

Server-path A/B on the real AOT images (`-ub 8,512`, i.e. reorder fired - see section 3b):

| image | pp512 post-reorder |
| --- | ---: |
| `:meat2` (no dequant fix) | 663.37 +/- 1.60 |
| **`:meat3` (deployed)** | **765.24 +/- 1.90** (**+15.4%**) |

The JIT build predicted 768.58 vs 765.24 measured on AOT - again within ~0.5%.

Verified on the deployed image, AOT vs AOT (**prefill-only numbers - the server's real prefill
is ~28% lower because the SoA reorder fires; see section 3b**):

| test | `:meat` (old) | `:meat2` (new) | |
| --- | ---: | ---: | ---: |
| pp512 | 848.62 +/- 1.98 | **916.73 +/- 2.85** | **+8.0%** |
| pp2048 | 812.43 +/- 1.23 | **876.51 +/- 1.18** | **+7.9%** |
| tg128 | 41.58 +/- 0.07 | 41.94 +/- 0.10 | +0.9% |

Cold start **2.95s** to model-loaded-and-listening, which also proves AOT is in use - a silent
`spir64` JIT fallback would take ~90s. Startup reports `GGML_SYCL_F16: yes`,
`GGML_SYCL_DNNL: yes`, `GGML_SYCL_MMVQ_MAX_BATCH: 32`.

**JIT builds are a valid proxy for AOT perf.** The JIT A/B predicted pp512=916.83; AOT measured
916.73. Use a JIT build (minutes) for future A/Bs instead of a 40-120 min AOT rebuild.

`libdnnl.so.3` must be copied into the image at `/app/` (it is on `LD_LIBRARY_PATH`).
Do **not** work around a missing DNNL by setting `GGML_SYCL_DNN=OFF` - that silently disables
the F16 path, which is the single largest prefill win.

### B50 (bmg_g21) - validated on real hardware 2026-07-16

Host `screamer` (10.0.0.200, `ssh -i ~/.ssh/id_nullraptor nullraptor@10.0.0.200`): Arc Pro B50 at
84:00.0, 16304 MiB (16228 free), Xeon E5-2690 v4, Docker 29.1.3, Level Zero present, 1 Gb/s link.
It has **no model and no /mnt/ignite**, so image (~9 GB) and GGUF (6.7 GB) must be shipped:

    rsync -e "ssh -i ~/.ssh/id_nullraptor" <model.gguf> nullraptor@10.0.0.200:~/bonsai-models/
    docker save llama-cpp-intel:prism-concatfix | gzip -1 \
      | ssh -i ~/.ssh/id_nullraptor nullraptor@10.0.0.200 'gunzip | docker load'

Model transfer ~71s at ~96 MB/s; image ~6.5 min (gzip -1 is the bottleneck, not the link).

**The dual-arch AOT works on real B50 hardware.** Device reports Level Zero **20.1.0**, matching
the `bmg-g21` IP version from `ocloc ids`. 9 bench runs completed in 58s wall *including* model
load - JIT alone would cost ~90s, so `bmg_g21` AOT code is genuinely in use.

| test | B50 | B70 | B50/B70 |
| --- | ---: | ---: | ---: |
| pp512 | 382.74 +/- 1.18 | 916.73 | 42% |
| pp2048 | 362.01 +/- 0.97 | 876.51 | 41% |
| tg128 | **20.39 +/- 0.11** | 41.94 | **49%** |

**The TG ratio corroborates the roofline independently.** B50 spec bandwidth (~224 GB/s) over
B70's measured ~460 GB/s is 0.487; measured TG ratio is 0.486. B50 does 20.39 x ~7.45 GB
= ~152 GB/s, i.e. **~68% of its own ceiling - identical efficiency to B70**. Both cards are
bandwidth-bound on weight reads, and the MMVQ kernel is equally (in)efficient on both. A win in
MMVQ or an XMX Q2_0 GEMM should therefore transfer to B50 proportionally.

**Context:** both `-c 131072` and `-c 65536` start and serve (health 200) with F16 KV. This
**contradicts the earlier arithmetic** in section 1 that called `-c 131072` "marginal to OOM".
Caveat: the server logs `common_init_result: fitting params to device memory ...`, so `-fit on`
may be silently reducing the context - the effective `n_ctx` after fitting was not captured.
**RESOLVED 2026-07-16: B50 serves the full `n_ctx = 131072`, not reduced.** `-fit` prints its
message but leaves context intact - it fits because `n_parallel=4` with `kv_unified=true` shares
one ~8 GB KV cache across all 4 slots (not 8 GB x 4): 6.7 weights + 8 KV ~= 14.7 GB in 16.2 GB.
All 4 slots report `n_ctx = 131072`. Load took ~43s on the slower Xeon host (fit search + reorder).
(Note: only `prism-concatfix` is on screamer; `prism-dequantfix`/`:meat3` was not re-transferred.)

**Operational warning:** llama-server startup at large `-c` on B50 (KV alloc + `-fit` search +
warmup + Q2_0 SoA reorder) pegs the GPU for a sustained period and makes the host sluggish. Run
B50 containers with `--rm` and a foreground wait, never `docker run -d` with cleanup only at the
end of a loop - an interrupted script orphans the container and it keeps grinding.

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

**Non-contiguous concat work-group size (+7% prefill).** Found via the profile (section 4), which
put `concat` at 9.0% of prefill. The non-contiguous kernel was launched with local range
**(1,1,1)** - one work-item per work-group, each serially walking all `ne0` elements - while the
kernel body was already written for a multi-item group:

    for (int i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2))

That stride loop had been running with a stride of 1. CUDA launches the same kernel with
`CUDA_CONCAT_BLOCK_SIZE` (256) threads/block (`ggml/src/ggml-cuda/concat.cu:178`); the SYCL port
dropped it. Fix: launch `SYCL_CONCAT_BLOCK_SIZE` work-items per group, body unchanged.

| test | before (local=1) | after (local=256) | |
| --- | ---: | ---: | ---: |
| pp512 | 857.65 +/- 0.86 | **916.83 +/- 3.03** | **+6.9%** |
| pp2048 | 814.70 +/- 0.70 | **875.19 +/- 2.37** | **+7.4%** |
| tg128 | 41.95 +/- 0.08 | 41.97 +/- 0.10 | unchanged |

TG is unaffected: at `n_tokens=1` the transposed operand still satisfies `ggml_is_contiguous`
(`ne[0]==1` short-circuits the check), so TG always took the contiguous path. PP diverges because
`ggml_transpose` on `qkv_mixed` is non-contiguous once `n_tokens > 1`
(`src/models/delta-net-base.cpp:473`, delta-net prepending conv state).

**Generic backend fix, not model-specific** - it affects every CONCAT with a non-contiguous
operand on SYCL (deepseek2, mamba, kimi-linear, rwkv6qwen2, ...). Good upstream candidate for
ggml-org. `test-backend-ops -o CONCAT -b SYCL0` passes, including the v=2/v=3 non-contiguous
variants.

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

**Audit for more `local=(1,1,1)` launches (2026-07-16): nothing else to fix.** After the concat
win, swept the backend for the same pattern. `cpy.cpp` has 11 launches with local range 1
(`ggml_cpy_f32_q8_0_sycl` and friends), but **CUDA does the same** - all 11 equivalents launch
`<<<num_blocks, 1, 0, stream>>>` (`ggml/src/ggml-cuda/cpy.cu:252`). That is upstream parity, not
a porting slip: those kernels are written for one work-item per quantization block, whereas
concat's body had a `get_local_range(2)` stride loop that its launch never fed.

Still theoretically suboptimal in *both* backends (1 work-item per group wastes the SIMD width),
but it is off our path - the Q8_0 copies serve KV quantization, which we rejected - and changing
it would diverge from upstream without a measurement to justify it.

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

## 3b. THE SoA REORDER PENALISES PREFILL BY 28% (found 2026-07-16)

**Read this before trusting any pp number in this document, including our own.**

`llama-bench -p 512` runs prefill *before* any decode, so the Q2_0 SoA reorder never fires and
prefill uses the **AoS** dequant kernel. A real server always decodes first, which runs MMVQ,
which triggers `opt_for_reorder()` - and from then on every prefill dequantizes from **SoA**
(`dequantize_row_q2_0_sycl_reorder`), which is much slower.

| scenario | pp512 |
| --- | ---: |
| prefill only, reorder never fires (what we have been measuring) | **916.23 +/- 4.59** |
| MMVQ runs first so reorder fires, then prefill (**what the server does**) | **661.99 +/- 0.79** |
| `llama-bench -p 512 -n 128` | 912.96 - misleading, it runs pp512 *before* tg128 |

**So the deployed server's prefill was ~662 t/s, not 916.** This is a real part of the
bench-vs-server gap noted in section 2. **Now ~769** after the dequant fix below; the deployed
`:meat2` image predates that fix and still runs at ~662.

Reproduce it without a server: `-ub 8,512` makes ub=8 take the MMVQ path (8 <= cap), firing the
reorder, and the following ub=512 then measures the post-reorder prefill.

### The trade-off - the reorder is still correct, keep it on

| | reorder ON (default) | reorder OFF (`GGML_SYCL_DISABLE_OPT=1`) |
| --- | ---: | ---: |
| tg128 | **42.26 +/- 0.13** | 15.71 +/- 0.03 |
| pp512 (post-reorder) | 661.76 +/- 0.68 | **919.46 +/- 8.41** |
| pp512 @ ub=8 (MMVQ) | 171.14 +/- 0.44 | 79.20 +/- 0.10 |

TG gains **2.7x** from the reorder; prefill loses 28%. Keep it enabled. But this makes the SoA
dequant kernel a concrete, well-scoped target:

### Partly fixed 2026-07-16: +16% on the server prefill path

`dequantize_block_q2_0_reorder` gave each work-item a whole QK2_0 block and looped 128 elements
serially, so adjacent lanes wrote addresses 128 elements apart - every store in a sub-group on its
own cache line. The AoS path uses the usual one-element-per-lane pattern, which is why it was
faster despite the worse layout.

Rewrote it as one qs byte (4 elements) per work-item, adjacent lanes on adjacent bytes:

| variant | pp512 post-reorder |
| --- | ---: |
| original (1 block/lane, serial 128-loop) | 661.99 +/- 0.79 |
| 1 element/lane (coalesced) | 763.27 +/- 2.50 |
| **1 qs byte/lane (current)** | **768.58 +/- 1.07** (**+16.1%**) |
| AoS ceiling (reorder not fired) | 916 |

tg128 unchanged (42.23). The jump is all from coalescing; going 1 -> 4 elements per lane added
only +0.7%, so load count was not the issue.

Correctness: verified by generating through the post-reorder prefill path (a 40x repeated long
prompt answers correctly; short factual prompts correct). **`test-backend-ops` does not cover this
kernel** - it never sets the reorder flag, so it only ever exercises the AoS path. The
`q2_K`/`q4_K` GET_ROWS failures on this branch are pre-existing tolerance noise (~2.6e-7 vs 1e-7),
unrelated; all `q2_0` cases pass.

**Still open - ~16% left (768 vs 916).** Profiling the post-reorder path shows the whole remaining
gap is still in this kernel; everything else matches the AoS profile:

| task | SoA (now) | AoS |
| --- | ---: | ---: |
| dequant | **0.494 s** | **0.267 s** |
| DNNL gemm | 0.375 s | 0.368 s |

**CONFIRMED 2026-07-16 - two-stream read is the cost.** Diagnostic: making the kernel read the
scale from the near (qs) region instead of the far d region - wrong output, timing only - jumped
pp512 from **769 to 850.50**, i.e. most of the remaining gap to the AoS 916. So `qs` at offset 0
and scales at offset `k/4` being two distant memory streams is the dominant remaining cost; AoS
keeps `d`+`qs` in one 34-byte `block_q2_0` struct (one stream).

**Not a kernel-only fix, and it has a real tension.** The SoA layout is not incidental - it exists
*because* it makes MMVQ fast: consecutive blocks' `qs` are contiguous, which is what the dp4a
`reorder_vec_dot` wants, and MMVQ is worth **2.7x on TG**. AoS is better for dequant, SoA for
MMVQ; a single layout cannot win both unless scales are duplicated or interleaved at some cost.
Closing this fully means a layout change touching all four SoA sites (`reorder_qw_q2_0`,
`dequantize_block_q2_0_reorder`, `reorder_vec_dot_q_sycl<Q2_0>`, `block_q_t<Q2_0>` offsets in
quants.hpp) and must be validated to not regress the 2.7x TG. Scoped next-session work, not a
tweak. Tried and rejected as kernel-only: staging scales is pointless (they are tiny and already
broadcast-coalesced); the two-stream *latency*, not scale volume, is the issue.

Note this also means the section 4 profile *understated* dequant: it profiled pp512 alone, i.e.
the fast AoS kernel, at 23.7% of prefill. On the server path dequant is a larger share still.

## 3c. RAISE n_ubatch - +35% prefill, no code, no downside (found 2026-07-16)

`n_ubatch` defaults to **512** and nobody had ever swept it. It is worth ~35-39% of prefill on
long prompts.

**Why:** dequantization cost is **per ubatch** - each ubatch dequantizes all 26.9B weights to F16
(section 3b/4). A 2048-token prompt at `-ub 512` therefore dequantizes the entire model **four
times**; at `-ub 2048`, once. Larger ubatch amortizes the dominant cost of the prefill path.

All measured on the server path (reorder fired via a leading `-ub 8`), B70, neighbour
interference checked = 0:

| n_ubatch | pp2048 | pp4096 |
| ---: | ---: | ---: |
| 8 | 169.40 | 167.87 |
| 256 | 521.42 | - |
| **512 (default)** | **734.56** | **686.55** |
| 1024 | 911.87 | - |
| **2048** | **1019.44** (+39%) | **926.90** (+35%) |
| 4096 | - | **956.90** (+39%) |

**No downside on short prompts:** pp512 measures 769.36 at `-ub 512` vs 767.38 at `-ub 2048` -
identical, because the ubatch is naturally capped by the actual prompt length. So a large `-ub`
only ever helps.

VRAM: both `-b 512 -ub 512` and `-b 2048 -ub 2048` start fine at `-c 131072` on B70.

**Deployed 2026-07-16** with `-b 2048 -ub 2048`. Real server log: an 819-token prompt evaluates at
**714.55 t/s**.

**Concurrency caveat MEASURED 2026-07-16 - no downside, `-ub 2048` is better here too.** 1
big-prompt user (~1800 tok) + 3 small interactive users fired concurrently (`-np 4`):

| metric | -ub 512 | -ub 2048 |
| --- | ---: | ---: |
| small-user worst latency | 4.56 s | **3.70 s** |
| small-user TTFT | 3520 ms | **2511 ms** |
| big-prompt total | 5.08 s | **4.19 s** |

The coarser-scheduling worry was backwards: the big prompt prefills 35% faster at `-ub 2048`, so
it clears the queue sooner and the small users waiting behind it are serviced sooner. That beats
the finer interleaving of `-ub 512`. Keep `-ub 2048`.

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

- ~~**The crossover is well above 32.**~~ **MEASURED 2026-07-16: the crossover is ~32-64, so the
  cap of 32 is already about right. Do not extend `switch_ncols`.** Clean numbers (3 reps, +/-0.03,
  verified free of neighbour-container interference), all with the reorder fired:

  | ubatch | MMVQ | oneDNN |
  | ---: | ---: | ---: |
  | 8 | 171.3 | (capped) |
  | 16 | 192.6 | - |
  | 32 | **207.3** | 98.9 |
  | 64 | (not instantiated) | **137.8** |
  | 128 | - | 261.8 |
  | 256 | - | 701.3 |

  MMVQ scales weakly (171 -> 193 -> 207), so an extrapolated MMVQ@64 ~= 215-220 vs oneDNN@64 = 138
  - MMVQ would still win at 64, and even at 128 (261.8) it would be close. **Revisit:** extending
  to 64 looks worth ~1.5x in the 33-64 band. It was deprioritized only because the band is
  narrower than the 9-32 one and costs 32 more instantiations per type. Not a settled "no".

  **DECISION 2026-07-16: do NOT implement now.** Checked when batch 33-64 actually occurs in this
  deployment: never. `n_parallel=4` caps the decode batch at 4, and prefill uses `-ub 2048` ->
  oneDNN. MMVQ only ever sees batch <=4 here, so both the 8->32 cap *and* any 33-64 extension are
  inert for this config - they matter only with high `n_parallel`, speculative decoding, or a small
  ubatch. The 8->32 change stays (correct and free when the band is hit), but extending
  `switch_ncols` to 64 (~32 x 18 = ~576 added instantiations, big build-time and .so cost) is not
  justified until a workload actually produces batches >32.
- **ub=24 (168) is below ub=16 (190) and ub=32 (204).** Non-monotonic; likely a tail/occupancy
  effect on a non-power-of-2 ncols. Minor, but it means the MMVQ ncols kernels are not uniformly
  tuned.

Caveat: `llama-cpp-sycl` (spans `level_zero:0;1`) was resident but idle during these runs, so it
held VRAM on the device. It affected both arms equally.

### Where does the time actually go? - ANSWERED 2026-07-16

Profiled with `vtune -collect xpu-offload` (note: `gpu-offload` is deprecated in VTune 2026.3).
`dev.i915.perf_stream_paranoid` was already 0. Host VTune:
`/opt/intel/oneapi/vtune/latest/bin64/vtune`.

**The delta-net is NOT the bottleneck. It was the wrong target.** Prior sessions (and the first
draft of these notes) assumed 48/64 layers meant delta-net dominated. It does not - they are
cheap layers.

#### PP512 (820 t/s under VTune vs 861 native, so overhead is low and shares are trustworthy)

Excluding the one-time load memcpy, compute totals ~1.13 s:

| Task | Time (s) | Share |
| --- | ---: | ---: |
| DNNL `gemm_kernel` | 0.368 | **32.6%** |
| **`dequantize_block_sycl<128,1,...>`** | **0.267** | **23.7%** |
| `gated_delta_net` | 0.108 | 9.6% |
| `concat` | 0.102 | 9.0% |
| `convert_unary_nc` | 0.065 | 5.8% |
| `flash_attn_tile` | 0.058 | 5.1% |
| `ssm_conv` | 0.038 | 3.4% |
| `swiglu` | 0.027 | 2.4% |

**Dequantization costs 73% as much as the matmul it feeds; together 56% of prefill.**

The dequant+DNNL path expands all 26.9B params to F16 on *every ubatch*: ~54 GB written and
re-read, vs 7.15 GB if the weights were consumed quantized. That is ~115 GB of traffic per
512-token prefill; at the measured 460 GB/s ceiling that is a ~250 ms floor, and pp512 takes
~624 ms. Prefill sits at roughly 19% of XMX peak - **it is not compute-bound, it is
bandwidth-bound on dequantizing its own weights.**

#### TG128 (19 t/s under VTune vs 42 native - heavy overhead, read as shares only)

| Task | Time (s) | Note |
| --- | ---: | --- |
| `reorder_mul_mat_vec_q2_0` (8 entries) | ~2.05 | dominant |
| `zeCommandListAppendMemoryCopy` | 1.058 | mostly one-time load, see below |
| `get_rows_sycl_float` | 0.127 | |
| `flash_attn_tile` | 0.066 | ~3% |
| `gated_delta_net` | 0.045 | **~2%** |

TG is MMVQ-bound, i.e. weight-read bound. Expected for memory-bound decode.

#### The memcpy line is benign - do not chase it

`zeCommandListAppendMemoryCopy` reports 33.2 GB "Host-to-Device" on a 6.7 GiB model, which looks
alarming. It is not. Scaling `-n` 8/64/128 fits exactly:

- instances = 1403 fixed + **56 per token** (56 ~= 48 delta-net layers + 8)
- bytes = 13.77 GB fixed + **152 MB per token**
- time = 1.019 / 1.036 / 1.058 s - **nearly flat**

The flat time is the tell: +18 GB in +0.039 s = **~460 GB/s**, i.e. device-local VRAM bandwidth,
not PCIe. VTune labels it H2D because the source is a host-USM pointer, but the data is
device-resident. The 152 MB/token is the recurrent state (48 x 3.1 MB = 149 MB, matching the
149.626 MiB checkpoint). At 460 GB/s that is 0.33 ms against a 24 ms/token budget: ~1.4%.
The 13.77 GB fixed baseline is model load + reorder round-trips.

#### Measured bandwidth ceiling (useful byproduct)

That memcpy rate gives B70's real achievable bandwidth: **~460 GB/s**.

TG reads ~7.15 GB weights + ~0.3 GB state per token. At 42 t/s that is **~313 GB/s = ~68% of
ceiling**. Headroom ~1.47x, i.e. a perfect MMVQ would reach ~62 t/s. 68% is already decent for a
quantized GEMV, so this is a grind, not a windfall.

### Context checkpoints

~150 MiB per checkpoint (the recurrent state of 48 delta-net layers: 16 groups x 128 state x 384
head-dim x 4 B x 48 layers ~= 151 MiB, matching the observed 149.626 MiB), at `min spacing = 256`
tokens. On a 4096-token prompt that is ~16 device-to-host copies, ~2.4 GiB of PCIe traffic during
prefill. llama-bench does none of this.

**Do not simply disable them.** Recurrent state cannot be rewound to an arbitrary position, so on
a prompt-cache miss the only alternative to restoring a checkpoint is reprocessing the whole
prompt. They are load-bearing for multi-turn TTFT. Tunables: `-ctxcp N`
(`LLAMA_ARG_CTX_CHECKPOINTS`, default 32) and `-cms N` (`--checkpoint-min-step`, default 256).

**MEASURED 2026-07-16: checkpoint settings are throughput-neutral. Keep the defaults.** A 600-token
generation at default / off / reduced:

| setting | gen t/s |
| --- | ---: |
| default (32 cp, step 256) | 41.5 |
| checkpoints OFF (`-ctxcp 0`) | 41.3 |
| reduced (8 cp, step 512) | 41.2 |

Within ~0.3%. Each 150 MB copy is ~0.33 ms at 460 GB/s and only ~2-3 fire per 600-token
generation, so they do not slow decode. The "pure overhead" worry above (the 2.4 GiB figure) is
real in bytes but overlaps compute and costs no measurable time. This run also pinned steady-state
server TG at **41.5 t/s** (matching llama-bench); the 25.6 t/s in the very first investigation was
a 5-token cold sample, not a real gap.

---

## 5. Next steps

P0 - free, no rebuild:

1. ~~`-ctk q8_0 -ctv q8_0`.~~ DONE - **rejected**, -41% TG at d=32768. Keep F16 KV. See section 3.
2. ~~Sweep `-cms` / `-ctxcp`.~~ **DONE - throughput-neutral, keep defaults (section 3b/4).**
3. Build a benchmark that mirrors the deployment (`-c`, `n_parallel`, graph, prompt cache) and
   re-baseline. The 861/42.1 numbers are not what production does.

P1 - cheap, resolves open questions:

4. ~~A/B `GGML_SYCL_MMVQ_MAX_BATCH` 8 vs 32.~~ DONE - 32 wins ~2x at batch 16-32. Next: find the
   real MMVQ/oneDNN crossover above 32 (needs `*_switch_ncols` instantiated past 32).
5. ~~Sweep `-ub` / `-b`.~~ **DONE - `-ub 2048` is worth +35-39% prefill, deployed. See section 3c.**
   Concurrent-user latency measured: `-ub 2048` is better under load too (section 3c). No open item.
6. Get a B50 baseline on the dual-arch build.
7. Establish the bandwidth ceiling. **This decides whether P2/P3 are worth doing at all.**

P2 - profile:

8. ~~VTune the current build.~~ **DONE** - see section 4. It invalidated much of the old P3 list.
9. ~~Check the output head.~~ **DONE - no win.** In the TG profile it is 0.079s (~4% of MMVQ
   time), proportionate to its 4.7% share of weight bytes, and already on the MMVQ reorder fast
   path. Not a hotspot.

P3 - reprioritized by the profile (2026-07-16):

10. **Native Q2_0 XMX GEMM for prefill - the biggest lever by a wide margin.** Dequantization is
    23.7% of prefill on its own, and the dequant+DNNL path moves ~115 GB per 512-token prefill
    versus ~7.15 GB if weights were consumed quantized. A `joint_matrix`/XMX Q2_0 GEMM that
    dequantizes **in-register** and feeds XMX directly would delete ~54 GB of traffic per ubatch.
    Prefill is at ~19% of XMX peak because it is bandwidth-bound on its own dequant, so the
    ceiling here is large.

    This is *not* a repeat of the failed MMQ attempt. That kernel was **dp4a**, which cannot beat
    XMX; the 153 vs 861 t/s result stands and is not evidence against this. It is the same
    mechanism that makes MMVQ beat oneDNN 2x at batch <= 32: skipping the dequant entirely.
    Note there is currently **no `joint_matrix` use anywhere in the SYCL backend**
    (`grep joint_matrix ggml/src/ggml-sycl/` is empty), and `common.hpp:105` admits
    `SYCL_USE_XMX` is "not used for XMX really" - it is only a dispatch gate. This is greenfield
    and a large job, but it is where the prefill time actually is.

11. ~~**`concat` is 9.0% of prefill.**~~ **DONE - fixed, +7% prefill.** Was a launch-config bug:
    local range (1,1,1) instead of 256. See section 3. Upstream candidate.
    **Re-profiled 2026-07-16: concat is now <1% (below 0.014s), off the top-10. Fully closed.**

12. **Chunked gated-delta-net: DEPRIORITIZED.** Worth <= 9.6% of PP and ~2% of TG. The earlier
    claim that this was the "highest ceiling" item was wrong - it was based on layer count
    (48 of 64) without measurement. They are cheap layers. The CUDA TODO
    (`gated_delta_net.cu:183`) still stands and is still unimplemented in both backends, but the
    payoff is bounded at ~10% of prefill.

13. **XMX flash attention: DEPRIORITIZED.** `flash_attn_tile` is 5.1% of PP and ~3% of TG, so the
    `fattn.cpp:192` TODO caps out around 5%. Not worth the large job. (It would, separately, be
    the enabler for quantized KV - see section 3 - but that is a memory-capacity argument, not a
    speed one.)

14. **MMVQ efficiency for TG.** ~68% of the measured 460 GB/s ceiling; a perfect kernel reaches
    ~62 t/s vs 42 today. Real but a grind, and 68% is already respectable for a quantized GEMV.
    **Checked 2026-07-16: `GGML_SYCL_MMV_Y=1` is already optimal (2->41.8, 4->41.0 t/s). No cheap
    win; the 2-bit->byte dp4a expansion is inherent ALU overhead, so real gains need a different
    kernel (overlaps the XMX GEMM, item 10/deferred).**

15. Do **not** restart dp4a MMQ. If any GEMM work happens, it is item 10 (XMX), not dp4a. If
    someone does revisit dp4a anyway, fix `VDR_Q2_0_Q8_1_MMQ` to 4 first.

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

## 6. Item 1 - native Q2_0 XMX GEMM (IN PROGRESS 2026-07-16)

Goal: replace dequant+oneDNN (which writes ~54 GB of F16 per ubatch) with an XMX int8 GEMM that
consumes Q2_0 directly. Distinct from the failed dp4a MMQ: dp4a is not tensor-core; XMX int8 DPAS
has ~2x fp16 throughput, so "dp4a can't beat oneDNN" does not apply.

### Feasibility - DONE

int8 `joint_matrix` (XMX DPAS) compiles AOT for bmg-g31 and runs correctly on B70. Probe in
`docs/xmx-probe/`. Learned: B must be VNNI-packed (`[(k/4)*N*4 + n*4 + (k%4)]`, stride N*4);
`layout::ext_intel_packed`; A/B must share element type; M8 N16 K32 tile works.

### Design

GEMM: `C[N_out, M_tok] = W[N_out, K] . X[K, M_tok]`, W=Q2_0, X=Q8_1 (activations already quantized
to int8+scale for MMVQ).

Key simplification vs MMVQ's bias trick: **expand Q2_0 to int8 `(raw - 1)` in {-1,0,1,2}**. int8
XMX handles signed values, so `sum((raw_w-1)*raw_x)` is computed directly - no separate
`-sum(raw_x)` bias term. Activations stay raw int8.

Scaling: XMX accumulates int32. W scale `d_w` is per 128 K (per N-row); X scale `d_x` is per 32 K
(per M-col). Accumulate int32 over a 32-K sub-block (both scales constant across it), then
`C[m,n] += (d_x[m] * d_w[n]) * acc[m,n]`. The scale is an outer product over the tile - the main
implementation cost, same problem CUDA `mmq.cuh` Q2_0 already solves (structural reference).

### Plan (incremental, each gated on correctness vs CPU ref, none touching live dispatch)

1. [DONE] int8 XMX probe (`docs/xmx-probe/jm_int8_probe.cpp`).
2a. [DONE] Q2_0 expand + VNNI + XMX int32 accumulation, host scaling, vs CPU ref - PASS maxerr 0
    (`docs/xmx-probe/q2_int32_accum.cpp`).
2b. [DONE] Fused single kernel: XMX -> SLM int32 -> scale by dx per sub-block -> dw multiply, vs
    CPU ref - PASS maxerr 0 (`docs/xmx-probe/q2_fused_slm.cpp`). The full kernel math and the SLM
    scaling pattern are proven.
3. [DONE - core mechanics] On-device expand from real `block_q2_0`/`block_q8_1`, K-loop over
   multiple blocks with per-block `d_w` folded into the SLM accumulation - PASS maxerr ~0.001
   (`docs/xmx-probe/q2_ondevice_kloop.cpp`, K=256). Decided the design fork: **feed from AoS
   `block_q2_0` directly** - XMX prefill and MMVQ decode are separate dispatch paths, so the XMX
   kernel reads AoS and leaves the SoA reorder (which MMVQ needs) untouched. Cleaner split, and
   AoS keeps `d`+`qs` in one stream for the expand.
   Remaining for a full kernel: M/N tiling for arbitrary dims (standard GEMM grid + boundary
   handling) - no novel mechanics left.
4. [NEXT, supervised] Wire into `ggml_sycl_op_mul_mat_q` behind an env flag, off by default; A/B
   vs oneDNN on the server prefill path. Promote only if it beats ~916 (AoS ceiling) and stays
   correct. Touches live dispatch - do with supervision.

**Status: every novel/risky mechanic of the XMX Q2_0 GEMM is proven with a passing standalone
test. What is left is tiling boilerplate + the live-dispatch A/B.**

Note: `joint_matrix_apply` in oneAPI 2026.1 exposes element values only, not coordinates, so
per-(m,n) scaling must go through SLM (validated in 2b), not an in-register apply.

Risk: the per-32-K outer-product scaling may erode the XMX throughput advantage. Item 2's finding
(prefill is dequant-traffic-bound, not compute-bound) is what makes this worth it - deleting the
54 GB F16 write is the win even if the GEMM itself is only par with oneDNN.

### Increment 3b - full tiled benchmark: correct at scale, naive perf is ~100x off

Built a full tiled kernel on a real Bonsai size (ffn_up: K=5120, N=17408, M=512).
`docs/xmx-probe/q2_tiled_bench_naive.cpp`.

- **Correctness: PASS at scale** (maxrel 0 on random spot-checks).
- **Perf: 55 ms/iter, ~0.43 GB/s effective weight reads (460 GB/s ceiling) - ~100x too slow.**
  Not memory-bound; catastrophically overhead-bound.

Root cause (diagnosed, not yet fixed):
1. **Redundant weight expansion.** Each weight block is expanded to int8+VNNI in SLM by *every*
   m-tile sub-group. With M=512 that is 64 re-expansions of the same weights. The expansion, not
   the XMX mad, dominates.
2. **Barrier storm.** 2 `group_barrier`s per 32-K sub-block x 160 sub-blocks per 8x16 tile. The
   arithmetic intensity is terrible: one 8x16x32 int8 mad (4096 MACs) per ~768 B of SLM shuffle
   plus 2 barriers.

Fix direction (the real GEMM engineering): each sub-group owns one N-tile and loops **all** M,
expanding each weight block **once** and reusing it across the M/8 m-tiles - amortizes expansion
64x. Needs a strip of accumulators in SLM (e.g. 8 m-tiles = 4 KB/sub-group), larger register
tiles, and minimal barriers. Standard for a production quantized XMX GEMM, but real work.

**GO/NO-GO verdict: still open.** The naive kernel does NOT prove the approach is slow - it proves
this structure is wrong. A properly-amortized kernel is required before comparing to oneDNN's ~916.
This is the multi-session engineering flagged from the start.

### Ceiling - MEASURED (corrected; an earlier draft said ~1.3x, that was too pessimistic)

Measured XMX peak on B70 (`docs/xmx-probe/xmx_peak_int8_vs_fp16.cpp`, register-resident mad chain,
no memory traffic):

| path | peak |
| --- | ---: |
| int8 DPAS | **358 TOPS** |
| fp16 DPAS | 183 TOPS |
| ratio | **1.95x** |

So int8 XMX genuinely runs ~2x fp16 on Battlemage. Ceiling from the prefill profile (section 4):
the XMX GEMM only touches the MUL_MAT part - dequant 23.7% + DNNL fp16 gemm 32.6% = **56% of
prefill**; the other ~44% (delta-net, attention, norms) is untouched (Amdahl bound).

- dequant (23.7%) -> fused away -> 0
- fp16 gemm (32.6%) -> int8 gemm at ~2x -> ~16%
- + activation int8 quant -> ~3%
- MUL_MAT portion: 56% -> ~19%; **prefill 100 -> ~63% -> ~1.6x** (916 -> ~1450 t/s equivalent).

**~1.6x prefill is the biggest single win available - larger than anything deployed tonight.** The
earlier "~1.3x" undercounted (forgot int8's 2x throughput). The catch is unchanged: realizing it
needs the int8 kernel to reach a good fraction of its 358 TOPS peak. The naive tiled kernel hit
1.7 TOPS = **0.5% of peak**; getting to ~50-70% (competitive with oneDNN's tuned fp16 efficiency)
is the hard, uncertain engineering.

**Honest synthesis: high reward (~1.6x prefill), high effort, uncertain (must reach good peak
fraction). Not "poor effort/reward" - it is the biggest lever, gated behind real kernel work.**
Correctness + feasibility + peak are all proven; the open question is purely whether a tuned kernel
reaches competitive efficiency. That is the next concrete step: engineer the amortized kernel
(weights expanded once per N-tile, reused across M; minimal barriers; larger tiles) and measure its
TOPS. If it clears ~150 TOPS it wins; if it stalls at tens of TOPS, stop.

### GO/NO-GO RESULT: NO-GO for this kernel design (2026-07-16)

Engineered the amortized kernel through five rounds, measuring TOPS at each:

| version | TOPS | note |
| --- | ---: | --- |
| naive tiled | 1.7 | weight re-expanded per m-tile |
| amortized (expand once/N-tile, A from global) | 5.0 | `q2_tiled_amortized.cpp` |
| + scale loads hoisted to SLM | 6.5 | |
| + in-register coord scaling (get_coord) | **11.3** | `q2_tiled_inreg_scale.cpp`, MSTRIP=4 |
| MSTRIP 8 / 16 | 3.9 / 4.8 | register spill - persistent float accumulators cap MSTRIP at 4 |

**Plateau at ~11.3 TOPS = ~3% of the 358 int8 peak.** The gate was ">=150 TOPS wins, tens = stop."
This is a stop.

Why it stalls: each M8xN16xK32 DPAS is surrounded by weight expansion + a barrier + an activation
load + per-32-K scale application. Q8_1's per-32 scaling forces the int32->float scale every
sub-block, which breaks the XMX pipeline; the tiny tile means overhead dominates; and the
persistent float accumulators needed to amortize the weight expansion cap the tile at MSTRIP=4
before register spill. All correct (maxrel 0 every version), just slow.

Comparison: this one ffn_up GEMM (91 GFLOP) takes 8 ms here; oneDNN's F16 GEMM at even 50% of its
183 TOPS peak does it in ~1 ms - **~8x faster, even carrying its dequant overhead**. So the fused
int8 kernel loses the GEMM despite skipping dequant. The ~1.6x ceiling is real but unreachable with
this straightforward design.

**Recommendation: NO-GO.** Reaching competitive XMX utilization would need expert-level GEMM
engineering (large register tiles without the per-sub-block scale stall - likely accumulating int32
across a full 128-K block and restructuring the scale, or a different activation quantization than
Q8_1's per-32), and even then success is uncertain against oneDNN's years of tuning. The proven
probes and this measured plateau are the value: they close the question cheaply instead of after a
multi-week kernel effort. **Item 1 is done - answered NO-GO, not abandoned.** Correctness,
feasibility, peak, and the practical ceiling are all measured and recorded.

### fp16-fused BUILT AND MEASURED - also NO-GO (2026-07-16)

After the int8 NO-GO, the review proposed fp16-fused (in-register Q2_0->fp16 weight dequant, fp16
activations, no per-sub-block scaling) as the lever that would dodge the int8 scale-stall. Built and
measured (`docs/xmx-probe/q2_fp16_fused.cpp`): correct (maxrel 1e-5), **12.5 TOPS, 7.28 ms**. Barely
faster than int8's 11.3 TOPS, ~10x off oneDNN's 124.7. **The scale-stall hypothesis was wrong** -
the real wall is the per-tile weight-expansion-to-SLM + barrier overhead of a hand-rolled tiled
GEMM, common to int8 and fp16. Does not beat the 1.11 ms current-path bar.

**Final: full NO-GO for hand-rolled fused GEMM (int8 or fp16).** The ~1.3-1.6x ceiling is real but
needs oneDNN-class GEMM engineering (register blocking, pipelining, no per-tile barriers) to reach -
not a reasonable ask against oneDNN itself. Standalone dequant is bandwidth-bound at the ceiling
(nothing left). Item 1 is fully closed.

## 7. dspark -60% ROOT CAUSE FOUND (2026-07-16)

**The dspark speculative-decode inversion (+34% CUDA vs -60% SYCL) is the Markov head running on
CPU, not the recurrent-state PCIe copy originally hypothesized.**

Chain of evidence:
- The prism server DOES engage dspark capture correctly (`server-context.cpp:1195`, PR #63) - the
  stale comment at `speculative.cpp:40` ("server path fails") predates prism's support. So dspark
  runs correctly; the -60% is a real perf result.
- The dspark Markov-head resample (`speculative.cpp:1350-1380`) has three compute paths: CUDA
  (`dspark-markov.cu`), BLAS (cblas_sgemv), and a naive host loop. **There is NO SYCL/GPU path.**
- The Markov weights are host-resident f32: `markov_head_a` [256 x 248320] BF16, expanded, and
  `markov_head_b` [256 x 248320] Q4_1 -> ~254 MB f32. Per drafted token the resample does a
  [248320 x 256].[256] GEMV over this 254 MB array + a full-vocab argmax. block_size=4.
- **Measured** the naive host GEMV (`markov_cost.cpp`): **113 ms per draft round** (4 tokens,
  254 MB, 9 GB/s single-thread). A target decode is ~24 ms/token. So each draft round adds ~113 ms
  of CPU Markov work - ~5x one decode step - which dwarfs any acceptance benefit. This exactly
  produces the -60%.

(BLAS/MKL multithreaded would be faster, ~30-40 GB/s -> ~30 ms/round, still > a decode step and
still net-negative. Only a GPU path fixes it. The deployed SYCL image was not built with
LLAMA_DSPARK_MARKOV_BLAS, so it uses the ~113 ms naive path.)

### Fix (tractable, unlike item 1)

Implement a **SYCL Markov-head resample**: the GEMV [n_vocab x rank].[rank] + argmax on the GPU.
Port `common/dspark-markov.cu` (the CUDA resample) to SYCL, or compute the GEMV via ggml on-device.
On GPU the 127 MB BF16 GEMV at ~460 GB/s is ~0.3 ms/token -> ~1.2 ms/round instead of 113 ms.
Unlike the XMX GEMM (item 1), this does NOT need to compete with a tuned library - it only needs to
be fast enough to stop being the bottleneck (~1 ms), which a straightforward memory-bound GEMV
kernel easily achieves. Expected result: dspark flips from -60% toward the CUDA-side +34%.

**FIX PROVEN STANDALONE 2026-07-16 (`docs/dspark-markov-sycl/markov_sycl_probe.cpp`).** Wrote the
SYCL resample (two-stage: per-work-group partial GEMV+argmax over vocab stripes, then final reduce,
`prev` chained on device). Correct - chained argmax matches the CPU reference exactly - and
**1.87 ms/round vs 113 ms host (60x)**, comfortably under the 24 ms decode step. Needed an in-order
queue for the `prev` chain (out-of-order raced the dPrev write/read; the ggml SYCL backend queue is
already in-order, so this is free in integration). The Markov head is no longer the bottleneck.

### ACTUAL FIX: one line, not a new kernel (2026-07-16)

Investigating the integration revealed a generic backend-agnostic Markov path ALREADY exists:
`llama_context::dspark_markov_resample` (`src/llama-context.cpp:2595`) builds a ggml graph
(get_rows(head_a) -> mul_mat(head_b) -> add(base) -> argmax, chained per block position) and runs
it on `model.dev_output()` via `ggml_backend_sched` - i.e. on SYCL when the model is on SYCL. It
falls back to the host path only when an op or type is unsupported.

Root cause of the fallback: `supported_head_type` listed F32/F16/BF16/Q4_0/Q5_0/Q8_0 but **NOT
Q4_1**, and the dspark model's `markov_head_b` is **Q4_1**. So the check failed and every round
took the 113 ms host path. All four graph ops ARE supported on SYCL (get_rows BF16, mul_mat Q4_1,
add, argmax - verified in ggml-sycl.cpp). **Fix = add Q4_1 (and Q5_1) to `supported_head_type`**
(`src/llama-context.cpp:2618`). No new kernel file, no separate device backend - the existing
generic path just needed the type allowed. My standalone SYCL probe (kept in
`docs/dspark-markov-sycl/`, 1.87 ms) is now only a reference/validation of the math, not the fix.

Rebuild is libllama + llama-server only (ggml-sycl `.so` unchanged), not the 40 min AOT. Then
deploy with dspark re-enabled and measure - expect the Markov head to run on GPU (~1 ms/round
instead of 113 ms) and dspark to flip from -60% toward the CUDA-side +34%.

Note the confidence head (`dspark.confidence_head`) and the drafter forward also run per round;
those are small (5376x1) and on the normal SYCL path. The Markov GEMV is the sole 254 MB CPU
outlier. Verify with a profile once the SYCL Markov path exists.

### dspark fix RESULT: markov bottleneck eliminated, dspark now neutral (not yet positive)

Rebuilt llama-server with the Q4_1 fix and tested end-to-end (drafter Q4_1, block_size=4, capture on
5 layers):

- **TG 17 -> 38.3 t/s (+125%)**, no host-fallback warning in the log -> the Markov head now runs on
  the SYCL GPU, not the 113 ms host path. **The specific bug is fixed.**
- But dspark 38.3 < 42 no-spec, i.e. still ~neutral-to-slightly-negative, because **draft acceptance
  measured only 0.27** (103/378), not the 84.9% the old notes claimed. At 27% acceptance most
  draft+verify work is wasted, and on a single GPU draft and verify serialize.

**Prompt-type sweep (2026-07-16, warm, 2 reps) - dspark is now net-positive, ENABLE it.** The
0.27-acceptance prose prompt was the worst case; real workloads vary widely:

| prompt | dspark (fixed) | vs ~42 no-spec | acceptance |
| --- | ---: | ---: | ---: |
| code | ~68 t/s | **+62%** | 0.69 |
| factual | ~49 t/s | +17% | 0.42 |
| chat/prose | ~41 t/s | neutral | 0.30 |

Textbook speculative-decode profile: large win on structured/code content, modest on factual,
harmless on prose. No host fallback in any run. (n_max is NOT tunable - it must equal the drafter's
block_size=4; `--spec-draft-n-max 2` fails at load. So there is no n_max sweep, only prompt type.)
The earlier single chat=2.8 t/s reading was a one-off glitch; warm it is 40.8/42.1.

**Verdict: the markov fix turns dspark from -60% (broken) into a net win on realistic workloads.
Enable it in the bonsai container.** The old "0.85 acceptance" claim was likely an even-more-
structured workload; the real range is 0.30-0.69, still a clear win where it matters. Single-GPU
draft+verify still serialize, but the numbers above already include that - it is a win anyway.
