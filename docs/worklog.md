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

## Open items
- aiter: blocked on AMD publishing the `v0.1.0-alpha2` wheel, or a from-source build attempt.
