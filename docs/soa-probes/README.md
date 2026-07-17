# SoA dequant / prefill kernel probes (2026-07-17)

Standalone benchmarks from the SoA-dequant investigation. Build:
`icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" -O2 <f>`
run with `ONEAPI_DEVICE_SELECTOR=level_zero:1`.

- `soa_layout_sweep.cpp` - proved dequant is LAYOUT-insensitive (flat across superblock size S), so
  the prefill gap was NOT the two-stream layout. Redirected the fix.
- `dequant_write_pattern.cpp` - found the real fix: 4 scalar half writes (0.801ms) vs 2x half2
  (0.374ms, 2.1x). This became the shipped SoA dequant fix (+29% server prefill).
- `getrows_float4.cpp` - get_rows gather: scalar (827 GB/s) vs float4 (1553 GB/s, 1.88x). Shipped as
  the F32 get_rows float4 fast path.
