# XMX int8 joint_matrix feasibility probe

Standalone probe for item 1 (native Q2_0 XMX GEMM). Confirms int8 `joint_matrix`
(XMX DPAS) works on Battlemage before any ggml integration.

## Result (2026-07-16, B70 / bmg-g31)

- Compiles AOT: `icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" -O2 jm_int8_probe.cpp`
- Runs on B70, `PASS: all 128 match` (int8 A x int8 B -> int32 C, M8 N16 K32).

## Key facts learned

- Matrix types: `layout::ext_intel_packed` (value 2), not a `packed` symbol.
- **B operand must be VNNI-packed**: `Bvnni[(k/4)*N*4 + n*4 + (k%4)] = B[k*N+n]`, loaded with stride `N*4`.
- A is `layout::row_major` stride K; accumulator int32; `joint_matrix_mad(sg, C, A, B, C)`.
- `joint_matrix_mad` requires A and B same element type (int8/int8 here).
- Tile M8 N16 K32 works; other Xe2 DPAS int8 shapes exist (query-types.hpp).
