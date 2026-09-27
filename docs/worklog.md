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

| 2026-09-26 | Same aiter run, continued past step 17 | `c1b5cf7` (`experiment/rocm714-ga-jax0110`) | **Hung at step 18.** 17 steps logged cleanly at ~15.5-16s/step, then no new step for 20+ min. Pod stayed `Running` (not crashed/exited) and `rocm-smi` showed all 8 GPUs busy the whole time (89-100% util, max boost clocks, normal temp/power). Killed after 20+ min with no resolution. ⚠️ Two claims in the original version of this row are corrected by the analysis row below: the earlier hang did **not** show idle GPUs, and "real compute work that never finished" does not follow from the util reading. |
| 2026-09-26 | Desk analysis of the step-18 stall — no GPU time spent, no run made (cluster has no GPU node) | n/a | **Leading guess (pathological batch 18) is contradicted by our own logs; the stall's signature matches the original Part 4 hang exactly.** Details in the section below. Instrumented diagnostic run prepared in `full-baseline.yaml`, ready to apply when a node is won. |
| 2026-09-26 | Streaming data loader: prefetch thread + stall watchdog + exact resume (`utils/data_utils.py`) | this branch | **Fixes an unbounded silent hang.** The old loop had no read timeout (a stalled Hub read blocks forever — no error, no EOF, no log line), no retry (an exception killed the run), and no prefetch (tokenizing 8,192 conversations ran between steps with the GPUs idle, which is why the step-22 shard boundary cost +16s on the critical path). Now a worker thread fills a depth-2 queue and the consumer times out after `BATCH_TIMEOUT_SECONDS=600`, reopening the stream at the exact position via `IterableDataset.state_dict()` — captured after each batch's rows are consumed and advanced only once the consumer takes the batch, so it resumes from what training saw rather than from how far the prefetch ran ahead. Verified locally against `datasets` 5.0.1 (consumed 3-7, resumed at 8) and with a generator that hangs forever at row 18: detected in 2.0s, resumed exactly, no replay and no gap. Side benefit for the diagnostic run — hypothesis 3 is now self-identifying: a data stall logs `Batch stream failed after N batches` and recovers, so a *silent* stall means it's device-side. |

## Step-18 stall: analysis before the next run (2026-09-26)

**Batch 18 is not pathological.** Data order is deterministic —
`utils/data_utils.py` streams with no shuffle, `skip_batches: 0`, fixed
seed — so every run consumes an identical batch 18. Two prior runs on two
different stacks did exactly that, at unremarkable speed:

| Run | step 17 | **step 18** | step 19 |
|---|---|---|---|
| `full-baseline-jax0112` (rocm10.0/jax0.11.2) | 57.3s, loss 2.1495 | **57.5s, loss 2.0137** | 57.2s, loss 2.0388 |
| `full-baseline-fullscale-jax082` (rocm7.2.4/jax0.8.2) | 66.7s, loss 2.1495 | **67.5s, loss 2.0137** | 67.6s, loss 2.0389 |

Identical losses to 4 dp, and step 18 sits at each run's median step time.
All shapes are static under `jax.jit`, so batch *contents* cannot change
which kernel aiter selects either. Demote this hypothesis.

**The signature is not new.** `gpu-hang-debugging-log.md` Part 4 describes
the original hang as *"GPUs sit at 100% utilization, unchanging, for the
entire hang"* — pod alive, no progress, threads in `kfd_wait_on_events`.
That is the step-18 signature. Root cause then was RCCL's MSCCL algorithm
on gfx950 (Part 7); the fix, `RCCL_MSCCL_ENABLE=0`, is **not** in the
current `full-baseline.yaml`. It was dropped as unnecessary, but that was
established on `rocm7.14.1`/`jax0.10.0` and `0.11.2` — this is a different
image (`therock-7.14` dev) and `jax==0.11.0`, so the ruling doesn't
transfer for free.

**`rocm-smi` "GPU use %" does not mean useful work** — it reports whether a
kernel is resident on the pipeline. A collective spin-waiting on a peer
pegs it at 100% at max clocks. The observation that temp/power looked
*normal* rather than near-TDP actually argues for spinning, not computing:
an 8192-sequence fwd+bwd at max clocks should draw close to TDP.

**Ranked hypotheses** (reasoned from source and logs; none yet observed
on hardware):
1. **Duplicate ROCm runtimes in one process.** The uv venv installs its own
   `_rocm_sdk_core` via `jax-rocm7-plugin`, while jax-aiter is compiled
   against the image's *system* ROCm — the reason we moved to the dev image
   at all. Two copies of `libamdhip64`/`librccl`/`libhsa` managing the same
   streams and signals is a textbook "runs fine, then wedges" bug, and
   `AITER_SYMBOL_VISIBLE=1` makes interposition likelier. Only genuinely
   new variable aiter introduced.
2. **The collective hang recurring, surfaced sooner by aiter.** ~64
   all-reduces/step × 18 steps ≈ 1,150 large collectives; aiter's 12x
   speedup raises the issue rate 12x, so a rare race arrives 12x sooner in
   wall-clock. Doesn't require aiter to be buggy.
3. **Host-side data stall.** `data_utils.py` is synchronous and unprefetched,
   tokenizing 128×64 = 8,192 conversations between steps. There is a real
   reproducible hitch in it: step 22 took 73.4s and 71.4s in two
   independent runs vs. ~57s elsewhere — almost certainly an HF streaming
   shard boundary. Ranked low only because it predicts *idle* GPUs.

**Also corrected:** `NCCL_DEBUG=INFO` was blamed for the 288MB
`full-baseline-run-v3.log`. The actual cause is the `TUNING` subsystem
specifically — 132,483 `pre`/`post`/`minNChannels` triplets, one set per
collective call. The whole `INFO` run in v2 is 6,219 lines. Use
`NCCL_DEBUG_SUBSYS=INIT,ENV`; MSCCL's lines are emitted at init. (RCCL keeps
NCCL's env-var names — there is no `RCCL_DEBUG`; it logs as
`NCCL INFO RCCL version : 2.27.7-HEAD`.)

## Open items
- aiter: from-source build works and delivers a large speedup; not yet merged toward `main` (still isolated on `experiment/rocm714-ga-jax0110`, and depends on the `jax==0.11.0`/`rocm7.14-GA` combo rather than main's `rocm10.0`/`jax==0.11.2`).
- **aiter stall at step 18, unresolved.** Don't trust this combo for an unattended long run until it's understood. Next step is the instrumented run below — deliberately *observation only*, no fix attempts, so it doesn't confound its own result.
- **Reliability before profiling** (agreed priority): get a run that starts and continues, then profile. Two things worth doing for reliability regardless of the diagnosis: `checkpoint.interval: 50` means a stall at step 18 loses the entire run — consider 10 while debugging; and the data loader has no read timeout or retry, so a hung HF streaming fetch freezes the job silently and forever.
- Tensor-core utilization at ~30%, up from ~19.3% pre-aiter but still well under 100% — headroom remains (e.g. `num_logit_iterations: 4` chunking, small effective per-matmul batch, non-ideal head_dim for aiter's fastest kernel). **Never actually profiled** — this rests on external monitoring plus reading the source. `jax.profiler.trace()` is the right tool and needs a small code change (the job clones from GitHub, so it needs a push). Deferred until a run is stable.
- The "64 all-reduces per step" figure comes from a **CPU simulation** (Part 14), never from the real GPU compiler. The diagnostic run dumps optimized HLO and prints a collective summary, which confirms or refutes it with real operand sizes — the input to deciding whether `perf/gradient-sync-shard-map` is worth landing.

## Ready to launch: instrumented diagnostic run

`full-baseline.yaml` carries an observation-only diagnostic block (remove
once resolved). Launch with `./docs/debug-logs/watch-aiter-stall-diag.sh`,
which waits for a GPU node, applies the job, then streams to
`docs/debug-logs/full-baseline-aiter-stall-diag.log`. Budget ~30-40 min of
node time for image pull, `uv sync` and the jax-aiter build before training
starts; the stall then arrives ~5 min in.

Only what reaches **stdout** survives — the pod is disposable — so summaries
are printed rather than only written to disk.

| Log output | Conclusion |
|---|---|
| `libamdhip64`/`librccl`/`libhsa` at **two paths** in the startup probe | Hypothesis 1 confirmed |
| py-spy frame in `next(iterator)` or a socket read, GPUs near idle | Hypothesis 3; rules out everything device-side |
| py-spy frame in `device_get`/PJRT, util 100% but power under TDP | Device-side spin → check the RCCL init log for MSCCL → hypothesis 2 |
| Stalls at exactly step 18 again | Deterministic, content/shape-dependent |
| Stalls at a different step, or not at all | A race → hypotheses 1/2 |

Deliberately **not** set: `RCCL_MSCCL_ENABLE=0`, `NCCL_ALGO=Ring`. Those are
fix attempts and belong in the run *after* this one.
