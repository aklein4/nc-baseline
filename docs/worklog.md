# Worklog

Concise experiment log. One row per run/change. Full narrative history
(pre-2026-09-26) is in `docs/gpu-hang-debugging-log.md` if needed.

| Date | Experiment | Commit | Result |
|---|---|---|---|
| 2026-09-26 | Bump flax floor to 0.12.7 (fixes removed `jax.core.get_opaque_trace_state` API on jax>=0.11) | `03521ba` (main) | Fixed — training gets past `initialize_params()` |
| 2026-09-26 | `rocm10.0`/`jax==0.11.2`, simplified `full-baseline.yaml` (plain `uv sync`, no manual override) | `4e223d4` (main) | Works, ~186s/step |
| 2026-09-26 | `shard_map` gradient-sync fix (1 psum/step instead of 64) | `16464c6` (`perf/gradient-sync-shard-map`) | Verified correct locally (CPU-sim, real model code, matches 1-device vs 8-device). **Not yet run on real GPU hardware.** |
| 2026-09-26 | `rocm7.14-GA`/`jax==0.11.0` (jax-aiter alpha2's exact target stack; aiter itself not installed — no public wheel exists) | `0efedcc` (`experiment/rocm714-ga-jax0110`) | Works, ~185s/step |
| 2026-09-26 | `rocm7.14.1`/`jax==0.10.0` (recreating the originally-documented-fast combo) | `c94d1d5`/`eb70ed0` (`experiment/rocm7141-jax0100-recheck`) | Works, ~186s/step |
| 2026-09-26 | Step-time investigation: is ~186s/step (vs. historical ~57s/step) caused by JAX version, node hardware, or interconnect bandwidth? | n/a | **All three ruled out.** Same JAX version (0.11.2) confirmed fast (57s) in a past wandb run and slow (186s) today. Same ~185s result on 3 different physical nodes. Isolated all-reduce bandwidth measured healthy (177 GB/s @ 1GB). Root cause unresolved — looks like something in the National Compute cluster environment itself, not diagnosable from inside a job. |
| 2026-09-26 | jax-aiter built from source (`release/v0.1.0-alpha2` branch, exact `jax==0.11.0`/`rocm7.14-GA` target) using AMD's prebuilt MHA/rmsnorm JIT libs instead of the multi-hour from-scratch build; `model.attention_backend=auto` enables it | `1fe5e2a` (`experiment/rocm714-ga-jax0110`) | **Big win.** Step time dropped from ~186s/step to **~15.5-16s/step steady-state — ~12x faster**, ~3.7x faster than the original ~57s/step baseline. Loss trajectory matches every prior run to 4 decimal places (same real workload, correct computation). Tensor-core utilization now ~30% (up from ~19.3% pre-aiter, per user's own monitoring). Repeating `[aiter WARNING] unsupported condition in fwd_v3` traced to source (`csrc/cpp_itfs/mha_fwd.cu`): our `head_dim=64` doesn't match the fastest ASM-v3 kernel's supported dims (128/128, 192/128, 256/256 only), so it falls back to AITER's general Composable Kernel (CK) path — confirmed correct, not a bug; the ~12x speedup is from the CK path alone, so there may be room for more if head_dim ever changes. |

## Open items
- aiter: from-source build works and delivers a large speedup; not yet merged toward `main` (still isolated on `experiment/rocm714-ga-jax0110`, and depends on the `jax==0.11.0`/`rocm7.14-GA` combo rather than main's `rocm10.0`/`jax==0.11.2`).
- Tensor-core utilization at ~30%, up from ~19.3% pre-aiter but still well under 100% — headroom remains (e.g. `num_logit_iterations: 4` chunking, small effective per-matmul batch, non-ideal head_dim for aiter's fastest kernel).
