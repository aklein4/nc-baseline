# CLAUDE.md

Working notes for this repo. Biased toward the things that have actually
cost time or money, not a description of the code — read the source for that.

## What this is

JAX/Flax training of a Llama-3.2-1B on episodic "Horizons" data, run as a
Kubernetes Job on National Compute's **MI355X (gfx950)** burst market.

- Entry point `src/train.py`, Hydra config `configs/`, default experiment
  `configs/horizon-lm-mi-8.yaml`.
- Batch shape is `(global_batch_size=128, cluster_length=64, max_length=1024)`
  — i.e. **8,192 conversations tokenized per step**, scanned per episode.
- `docs/worklog.md` is the experiment log (one row per run). `docs/national-compute.md`
  is the platform guide. `docs/gpu-hang-debugging-log.md` is the long-form history.

## National Compute: the expensive parts

**A job delete costs you the node.** Deleting hands it back to the market;
your next apply gets a *new* grant with its own minimum billed hold. Waiting
for reconciliation does **not** recover it. Measured 2026-09-27: delete →
re-apply immediately granted a different node; delete → wait 5 min → re-apply
granted the same different node again. Reclaim of the freed node then took
~7 minutes, idle and billable the whole time.

**Therefore: batch every change into one relaunch.** A pod template is
immutable, so any config change forces delete + re-apply. Prefer `kubectl exec`
on a live pod; consider reading experiment knobs from a ConfigMap so a new
experiment needs no new pod.

**Never pin a node.** Kueue injects its own selector on admission:

```
marketplace.nationalcompute.com/request: <this grant's uuid>
```

and the market relabels whichever node it assigns. Your `nodeSelector`/`nodeName`
ANDs with that, so if the market picks a different node the pod is
unschedulable forever (`0/N nodes ... didn't match Pod's node affinity/selector`).
Submit unpinned; two unpinned 8-GPU jobs get one node each.

**Diagnosing "waiting for market capacity"** — the console conflates two states:

```bash
kubectl get workload <name> -o jsonpath='{.status.admissionChecks}'
```

- `"waiting for the market"` → no node granted: shortage, **bid too low**, or balance too low.
- `"0 of N granted nodes ready"` → node granted, still booting. Nothing wrong.

**A $0 bid grants nothing, ever**, and says so explicitly in the admission
check message. This is the easiest way to have a job hang pending forever.

**Bare pods are gated too.** `mpod.kb.io` attaches a scheduling gate, and
`nodeName` cannot be set while a gate exists. Removing the gate bypasses
cluster admission control — don't, without an explicit decision from the user.

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

## Instrumentation: what works, what lies

| Signal | Verdict |
|---|---|
| `XLA_FLAGS=--help` | **Useless here.** Prints ~94 lines, mostly CPU flags, then dies "Flag parsing failed". A flag missing from it proves nothing — this wrongly rejected a valid flag. |
| XLA flag support | **Probe by launching** `XLA_FLAGS=<flag> python -c "import jax; jax.devices()"` in a throwaway process. An invalid flag aborts startup, so "it started" is definitive. |
| `rocm-smi` "GPU use %" | Only means a kernel is **resident**. A spin-waiting collective reads 100%. Use **power** and **memory R/W activity** as the honest signals — a stalled collective showed 100% use at ~25% of TDP with **0% memory activity**. |
| `/proc/<pid>/task/*/stack` | **Permission denied** in-container. A `kfd_wait_on_events` count of 0 measures nothing. |
| `py-spy dump` | Works, needs `SYS_PTRACE` in the pod securityContext. The best host-vs-device discriminator: a frame in `next(iterator)` means data, `device_get`/PJRT means device. |
| `NCCL_DEBUG_SUBSYS` | `TUNING` is the firehose (one triplet per collective call → 288MB logs), not `INFO` generally. Use `ALL,^TUNING`. |
| `AMD_LOG_LEVEL=1` | Reveals `UpdateStreams failed` from the HIP-graph path (~3,900/sec). Real signal, but ~1GB of log per hour. |
| `AITER_LOG_LEVEL=ERROR` | Silences aiter's benign `fwd_v3` warning (~570k lines/37 steps) without hiding real errors. Gated by `getenv` in `aiter_logger.h`; emitted from C++ to stderr, so Python logging cannot touch it. |

## Known issues and open questions

**The aiter stall (open).** With jax-aiter, runs hung at step 18, then step 11
— nondeterministic, so a race. Signature: pod alive, host blocked in
`jax.device_get` (`base_trainer.py`), GPUs at 100% "use" but ~25% TDP and **0%
memory traffic** — a collective spinning on a flag. Refuted: duplicate ROCm
runtimes (one `libamdhip64`/`librccl` only), and a data stall (py-spy proves
device-side). Two configs then ran 85+ and 128+ clean steps:
`RCCL_MSCCL_ENABLE=0`, and `--xla_gpu_enable_command_buffer=` (off).
**Attribution unresolved** — both passed, so either may be sufficient
(precedent: Part 7 found two independently-sufficient fixes for the earlier
hang). `--xla_gpu_nccl_termination_timeout_seconds` is *not* the fix: it only
terminates a stuck rendezvous, it cannot prevent one.
**The control run is: neither change, everything else identical.**

**`UpdateStreams failed`** comes from the HIP-graph/command-buffer path.
Disabling command buffers takes it to 0 at no throughput cost (12.47 vs
12.33 s/step), so command buffers buy nothing measurable here.

**Performance headroom.** `head_dim=64` misses aiter's fastest ASM-v3 kernel
(wants 128/128, 192/128, 256/256), so attention runs the Composable Kernel
fallback. Real HLO shows **22 static all-reduces, 14 inside a loop body**,
gradients reduced per-tensor in bf16 — Part 14's "64 per step" came from a CPU
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
