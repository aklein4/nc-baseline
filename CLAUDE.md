# CLAUDE.md

## What this is

JAX/Flax training of a Llama-3.2-1B on episodic "Horizons" data.

- Entry point `src/train.py`, Hydra config `configs/`, default experiment
  `configs/horizon-lm-mi-8.yaml`.
- Batch shape is `(global_batch_size=128, cluster_length=64, max_length=1024)`
  — i.e. **8,192 conversations tokenized per step**, scanned per episode.
- `docs/worklog.md` is the experiment log (one row per run). `docs/national-compute.md`
  is the platform guide. `docs/gpu-hang-debugging-log.md` is the long-form history.

## Job workflow

```bash
python3 docs/debug-logs/make-experiment-jobs.py   # derive per-experiment manifests
kubectl apply -f docs/debug-logs/job-<name>.yaml
./docs/debug-logs/watch-job.sh full-baseline-<name>   # attach + stream + capture as-run spec
```

`watch-job.sh` never applies or deletes, and captures the as-run spec from the
API on attach. **This matters**: the cluster clones `src/` from GitHub (so code
is pinned by commit) but the YAML is applied from your working tree. Those two
halves can diverge silently — commit before applying.

## Note

* Commit everything - .yaml files, .sh files, src files so we can trace what the exact config for a run was.
* Log everything - when running a job, capture device details, driver details, errors in communication etc before logging the runtime info.
* Think carefully about flags, versions and other things that might bite or be inconsistent when errors occur.

## Instrumentation: what works, what lies

| Signal | Verdict |
|---|---|
| `XLA_FLAGS=--help` | **Useless here.** Prints ~94 lines, mostly CPU flags, then dies "Flag parsing failed". A flag missing from it proves nothing — this wrongly rejected a valid flag. |
| XLA flag support | **Probe by launching** `XLA_FLAGS=<flag> python -c "import jax; jax.devices()"` in a throwaway process. An invalid flag aborts startup, so "it started" is definitive. |
| `rocm-smi` "GPU use %" | Only means a kernel is **resident**. A spin-waiting collective reads 100%. Use **power** and **memory R/W activity** as the honest signals — a stalled collective showed 100% use at ~25% of TDP with **0% memory activity**. |
| `/proc/<pid>/task/*/stack` | **Permission denied** in-container. A `kfd_wait_on_events` count of 0 measures nothing. |
| `py-spy dump` | Works, needs `SYS_PTRACE` in the pod securityContext. The best host-vs-device discriminator: a frame in `next(iterator)` means data, `device_get`/PJRT means device. |
| `NCCL_DEBUG_SUBSYS` | **`ALL,^TUNING` does NOT work** — this RCCL build ignores the `^` exclusion and logs everything anyway. Measured: it produced 26M tuning lines and 6.2M per-collective `AllReduce:` lines, ~8GB across three runs. Use an explicit allowlist: **`INIT,ENV`**. |
| Log volume | A run with `NCCL_DEBUG=INFO` + `AMD_LOG_LEVEL=1` writes GBs in minutes. Strip with `grep -v` on `UpdateStreams failed`, `unsupported condition in fwd_v3`, `NCCL INFO (pre\|post)-adjustment\|minNChannels\|Channel Tuning not applied\|TUNER/CsvTuner`, and `NCCL INFO AllReduce:` — that took 8.8GB → 472MB with every step line and heartbeat intact. Always verify step/heartbeat counts match before replacing the original. |
| `AMD_LOG_LEVEL=1` | Reveals `UpdateStreams failed` from the HIP-graph path (~3,900/sec). Real signal, but ~1GB of log per hour. |
| `AITER_LOG_LEVEL=ERROR` | Silences aiter's benign `fwd_v3` warning (~570k lines/37 steps) without hiding real errors. Gated by `getenv` in `aiter_logger.h`; emitted from C++ to stderr, so Python logging cannot touch it. |

## Known issues and open questions

**The aiter stall — RESOLVED as an MSCCL x command-buffer interaction.**
Signature: pod alive, host blocked in `jax.device_get` (`base_trainer.py`),
GPUs at 100% "use" but ~25% TDP and **0% memory traffic** — a collective
spinning on a flag, not computing.

| MSCCL | command buffers | runs | result |
|---|---|---|---|
| on | on | 3 | **stall @ 18, @ 11, @ 3** |
| **off** | on | 1 | clean, 85 steps |
| on | **off** | 1 | clean, 148 steps |

Neither change alone is sufficient to *cause* the failure, so the bug needs
both present and disabling either breaks the pairing. Consistent with
`UpdateStreams failed` coming from `hip_graph_internal.cpp` (the
command-buffer path) and stopping dead at every wedge. Same shape as Part 7,
where `NCCL_ALGO=Ring` and `RCCL_MSCCL_ENABLE=0` were each independently
sufficient.

**Use command-buffers-off** (`--xla_gpu_enable_command_buffer=`): longest
validated cell, ~1% step-time cost, and it removes the error storm.

Refuted along the way: duplicate ROCm runtimes (one `libamdhip64`/`librccl`
only); a data stall (py-spy proves device-side); heavy logging masking the
race (`NCCL_DEBUG_SUBSYS` identical in all four runs); and
`--xla_gpu_nccl_termination_timeout_seconds`, which only *terminates* a stuck
rendezvous and cannot prevent one.

Caveats: each clean cell is n=1, though at the observed ~0.09/step hazard,
surviving 85 and 148 steps has probability ~2e-4 and ~5e-7. The mechanism is
inferred from the grid, not seen in a trace. Durability beyond ~30 min untested.

**`UpdateStreams failed`** comes from the HIP-graph/command-buffer path.
Disabling command buffers takes it to 0 at no throughput cost (12.47 vs
12.33 s/step), so command buffers buy nothing measurable here.

**Performance headroom.** `head_dim=64` misses aiter's fastest ASM-v3 kernel
(wants 128/128, 192/128, 256/256), so attention runs the Composable Kernel
fallback.

**Collectives, measured from RCCL's own per-op logging over 148 real steps
(rank 0): 5,222 all-reduces per step, 159.8 GB/step.** Part 14's "64 per
step" was a CPU simulation and is wrong by ~80x.

| per step | size | GB/step | source |
|---|---|---|---|
| 3,093 | 32.0 MiB | 103.8 | `bf16[2048,8192]` MLP grads — 3/layer x 16 layers x 64 episodes = 3,072 |
| 1,031 | 20.0 MiB | 21.6 | `bf16[10485760]` fused buffer — 16 layers x 64 episodes |
| 65 | 501 MiB | 34.4 | `bf16[128256,2048]` embedding/lm_head grad — **these params are FROZEN** |
| 1,031 | 16 KiB | ~0 | small reductions |

Two things fall out. The counts confirm collectives fire **per layer per
episode inside the scan**, which is what `perf/gradient-sync-shard-map`
(unlanded) is meant to collapse. And **~21% of all collective traffic
(34.4 GB/step) is the embedding/lm_head gradient for parameters marked
FROZEN** — computed and all-reduced across 8 GPUs once per episode, then
thrown away by the optimizer. That looks like free savings independent of
the shard_map work.

At the 177 GB/s measured bus bandwidth, 159.8 GB/step is a floor of ~0.9s
against a 12.4s step, so collectives are significant but not dominant —
profile before assuming they are the bottleneck.

Older HLO note: the static module shows 22 all-reduces, 14 inside a loop
body; Part 14's estimate came from a CPU
simulation, not hardware. `perf/gradient-sync-shard-map` is the unlanded fix.
VRAM sits at 75-77%, so headroom for bigger batches is limited.
**Never profiled** — `jax.profiler.trace()` first (timeline), then
`rocprofiler-compute` only on whichever kernel dominates.

**Data loader.** `utils/data_utils.py` now prefetches on a worker thread with
a stall watchdog and exact resume via `IterableDataset.state_dict()`. This was
worth ~22% (15.5-16s → 12.1-12.4s per step) by moving tokenization off the
critical path. A stalled Hub read used to hang the job silently forever.

**Reclaim resilience is missing.** `backoffLimit: 0` with no `podFailurePolicy`
means a market reclaim kills the run outright. See the mitigation block in
`docs/national-compute.md` before any long unattended run.

## Conventions

- `*.log` is gitignored; run logs reach GBs. Never commit them.
- Generated manifests come from `make-experiment-jobs.py` — edit
  `full-baseline.yaml` and regenerate, don't hand-edit the outputs.
- Change **one variable per run**. This has been violated twice and both times
  made the result unattributable (see Part 6, and the 2026-09-27 runs).
