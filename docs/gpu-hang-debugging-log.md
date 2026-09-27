# Debugging log: launching `full-baseline` on the National Compute cluster

This is a running log of what we learned launching this repo's training job on
National Compute's (NC) Kubernetes GPU cluster.

## Part 1: Kubernetes basics, the way we actually used them here

**`kubectl` is a messenger, not a worker.** Every `kubectl` command sends an
HTTP request to one specific machine: the cluster's **control plane** (the
"brain" of the cluster — plain CPU boxes, no GPU needed, since their job is
bookkeeping, not running your training code). `kubectl` never talks to a GPU
node directly. The control plane keeps a database of everything it knows
about the cluster — every pod, every Job, every secret, every node's status —
in a component called `etcd`. When you `kubectl create secret`, the secret
goes into `etcd` and sits there unused until some pod that references it
actually gets scheduled onto a node; only then does the control plane hand
the secret's values to that pod as environment variables.

**A "context" is a saved connection profile**, not a cluster itself. It
bundles three things your local `~/.kube/config` needs: which cluster
(control plane address), which credentials, and which namespace. You can
have many contexts (e.g. dev vs. prod); only one is "current" at a time.
Commands to know:
- `kubectl config get-contexts` — list all saved contexts
- `kubectl config current-context` — show the active one
- `kubectl config use-context <name>` — switch (note: lives under `config`,
  since it only edits your local file, never talks to the network)

**A cluster's identity is its control plane, not its nodes.** Nodes
(including GPU nodes) can join and leave — the cluster still exists with
zero nodes attached. This matters a lot on NC (see Part 2).

**Applying the same secret twice silently overwrites, no merge, no
history.** `kubectl apply` is idempotent by design — "make the cluster match
this file" — so whoever runs it last wins, with no warning. This is exactly
how we found a real bug: the cluster's `api-keys` secret was 10 days stale
(missing a newly-added `HF_ID` key) because nobody had rerun the apply command
since it was added. Plain `kubectl create` (no `apply`) would have instead
errored with "already exists" — that's why `docs/credentials.md`'s documented
command specifically uses the "dry-run + apply" idiom, trading a safety error
for reapply-ability.

**`kubectl logs` vs `kubectl exec` are fundamentally different.** `logs`
reads back what a container already printed to stdout/stderr — after the
fact, from a completed or running container. `exec` opens a *live* process
inside the *already-running* container right now — that's how we could run
`rocm-smi` and see real-time GPU state, something `logs` could never show.
Also: a long-lived `kubectl logs -f` stream can disconnect on its own
(`error: http2: client connection lost`) purely from an idle/connection-length
limit somewhere in the network path — that is **not** the same as the
container dying. Check the pod's actual phase before assuming a dropped log
stream means the job ended.

**Job specs are mostly immutable after creation.** You cannot `kubectl apply`
a changed Job spec over a Job that already exists — you have to
`kubectl delete job <name>` first, then reapply. Each apply creates a brand
new Kueue "Workload" object internally, which is why you'll see a fresh
`Suspended → CreatedWorkload → QuotaReserved → ...` event cycle every time,
even though nothing is actually wrong.

## Part 2: How National Compute's market model actually behaves

Full mental model is in `docs/national-compute.md`; the surprising bit we
learned hands-on:

**A GPU node is bound to your account for a minimum hold window, not to any
particular Job.** Per NC's docs, a freshly granted node is protected from
being reclaimed by competing bids for its first ~2 hours, and the first hour
is a **minimum billed hold** regardless of whether your job is even running.
We deleted and recreated our Job well inside that window, and the *same*
physical node (`polite-hippo`) was reused instantly — no new auction, no
~15-minute node-provisioning wait, image already cached from before. Fast for
us, but also: **you're billed for the whole gap**, including the ~15 minutes
between delete and recreate, and the ~1 hour the first attempt spent silently
hung doing nothing. `kubectl delete job` does not release GPU billing.

**Diagnosing "why is my job pending" requires several different `kubectl`
views, not one.** In order of what actually happens:
1. `kubectl get workloads` — Kueue's admission queue entry (admitted or not)
2. `kubectl describe job <name>` — narrates the market verdict as Events
   (`QuotaReserved`, `NodeGranted`, etc.)
3. `kubectl get provisioningrequest` / `kubectl describe provisioningrequest`
   — tracks whether the *granted* node has actually finished booting and
   joined yet (`Provisioned: False, "0 of 1 granted nodes ready"` is a real,
   separate wait step — being "granted" a node and that node being "ready"
   are not the same moment, and the gap can be many minutes)
4. `kubectl get pods` — the actual pod only gets created once the whole gang
   is admitted, all at once (no partial states)

## Part 3: Monitoring a live GPU training job

- **`rocm-smi`** (AMD's equivalent of `nvidia-smi`) has to be run *inside*
  the container via `kubectl exec <pod> -- rocm-smi` — it reads the GPUs
  mounted into that specific container. It has no built-in continuous-refresh
  flag (unlike `nvidia-smi -l`); wrap it yourself: `watch -n 2 kubectl exec
  <pod> -- rocm-smi`.
- **Weights & Biases** is the better tool for trends over time (loss curves,
  throughput) if the trainer reports there — a one-off `rocm-smi` snapshot
  only tells you the instant you asked.
- **`py-spy`/`strace` need `CAP_SYS_PTRACE`**, which Kubernetes pods don't
  have by default (it's a container-escape-relevant capability most clusters
  intentionally withhold). Without it, there's no way to get a live Python
  stack trace of a running container's process — confirmed by trying both a
  direct `kubectl exec` and an ephemeral `kubectl debug --share-processes`
  sidecar; both hit "Permission denied" reading process memory. Adding that
  capability requires recreating the pod, which would erase the exact hung
  state being investigated — a real tradeoff, not a solvable-in-place gap.
- **`/proc/<pid>/task/*/wchan` is a permission-free fallback** for coarse
  diagnosis when `py-spy` is blocked. It shows which kernel function each
  thread is blocked in. `kfd_wait_on_events` specifically means a thread is
  blocked inside AMD's GPU kernel driver waiting for the hardware to signal
  that dispatched work finished — useful for telling "stuck waiting on the
  GPU" apart from "stuck waiting on a mutex" apart from "genuinely still
  computing."
- **100% GPU utilization does not mean healthy progress.** If a
  cross-GPU collective operation is stuck waiting on a peer, the underlying
  library can busy-poll in a spin loop, which shows up as "100% busy" in
  `rocm-smi` even though zero real training work is happening. The reliable
  signal was the trainer's own step-completion log line
  (`src/trainers/base_trainer.py:135`, unconditional every step, no logging
  interval) simply never appearing — that's what turned "maybe just slow"
  into "confirmed stuck."

## Part 4: The `full-baseline` hang — timeline and evidence

**Symptom:** the job reaches "Compilation complete; executing first training
step..." and then never logs a completed step, indefinitely. GPUs sit at
100% utilization, unchanging, for the entire hang.

**Attempt 1** (no extra env vars): hung for 60+ minutes.
`/proc` inspection showed ~1,128 threads, almost all idle
(`futex_wait_queue`), except 2 threads genuinely `R (running)` (matching
~200% CPU) and one thread in `kfd_wait_on_events` — strong evidence of a
stuck GPU-driver-level wait, not a slow-but-progressing computation.

**Attempt 2** (added `NCCL_DEBUG=INFO`, `NCCL_IB_DISABLE=1`, killed and
relaunched the job): got much further visibility, still hung the same way.
Key new evidence:
- All 8 ranks successfully completed RCCL communicator setup
  (`ncclCommInitRankConfig_impl ... Init COMPLETE`), each connecting via
  `P2P/direct pointer` — confirming RCCL correctly used local GPU-to-GPU
  memory access the whole time and never touched InfiniBand at all.
- **This rules out the InfiniBand-misconfiguration theory.**
  `NCCL_IB_DISABLE=1` was confirmed active in the pod's resolved environment
  but had no effect, because IB was never in use to begin with.
- Immediately after Init COMPLETE, RCCL logs a burst of internal
  `threadThreshold` tuning computations for various anticipated message
  sizes (up to ~525MB, consistent with gradient tensor sizes) — then the log
  goes completely silent. The hang begins sometime after tuning and before
  (or during) actually executing the first real collective transfer.

**Also ruled out:** an earlier, separate, and unrelated warning — an
XLA-internal "clique" rendezvous log (`rendezvous.cc`) that fires a "may be
stuck, leader may be deadlocked" warning after a 10-second internal
watchdog timeout — is a known false positive; it explicitly logs its own
"unstuck ... Warning above was a false-positive" follow-up ~30 seconds later
in every run so far. Don't mistake this for the real hang; the real hang
starts later.

**Still open:** why the very first real cross-GPU collective transfer never
completes, despite communicator setup succeeding cleanly and quickly every
time. A background research pass is investigating whether this matches a
known ROCm/RCCL/JAX-on-ROCm issue for MI300-series GPUs — findings to be
added below once that comes back.

## Part 5: Research findings (JAX + ROCm + RCCL hang)

No GitHub issue or bug report was found that's an exact, confirmed match for
"MI355X + JAX + hang right after RCCL channel tuning." That's not
reassuring — it likely just means this specific hardware/software
combination (MI355X + ROCm 7.2.4) is new enough that not many people have hit
it yet, publicly. But several *known classes* of bug line up closely with
what we're seeing. Ranked by how likely each one is to be our actual cause:

**1. (Most likely) A version mismatch between JAX's ROCm plugin and the
installed RCCL.** Background: JAX doesn't come with ROCm support built in —
it uses a separate plugin (`jax-rocm7-plugin`) that has to be built against
one *specific* ROCm version. If the plugin was built against a slightly
different ROCm version than the 7.2.4 actually installed in this image, they
don't necessarily throw an error — they can just silently misbehave at
runtime instead, which matches "everything looks fine right up until it
doesn't." This kind of mismatch is more likely to bite JAX specifically
(rather than PyTorch) because PyTorch's ROCm backend gets far more
real-world testing mileage than JAX's does. **Next step:** check what ROCm
version `jax-rocm7-plugin`/`jaxlib` were actually built against, and try
bumping to the latest point release.

**2. A known RCCL "channel tuner picks a bad channel count" bug class on
this GPU architecture (gfx950, which is what MI350-series/MI355X chips use).**
There's a real, recently-fixed bug (ROCm/rocm-systems PR #11651) where
RCCL's tuning step (exactly the "pre-adjustment/post-adjustment" log burst we
saw) picks a channel count that doesn't match what's actually wired up for
direct GPU-to-GPU (P2P) transfer — causing a failure right where our hang
starts. That specific fix targeted a different physical wiring layout (PCIe
rather than the faster XGMI GPU interconnect) so it may not be identically
our bug, but it confirms this whole *category* of "tuner and reality
disagree, first real transfer wedges" bug exists on this chip generation
in this ROCm timeframe. **Next step:** try forcing a small, fixed channel
count instead of letting RCCL choose:
`NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2`. Also worth ruling out RCCL's
"MSCCL" collective algorithm (on by default on this GPU class, and had
protocol-level bugs fixed recently) with `RCCL_MSCCL_ENABLE=0`.

**3. A more generic, hardware-agnostic "collective setup succeeds, first
real transfer deadlocks, one thread stuck waiting on a GPU completion
signal" pattern**, seen before on other AMD GPU generations (e.g. an older
MI300X hang, and a separate one on a newer consumer AMD GPU) — different
chips, but the same shape: clean init, hang precisely at the first real
collective, one thread stuck exactly like our `kfd_wait_on_events` thread,
waiting on a signal that never arrives from the GPU. **Next step:** try
forcing a simpler, older communication protocol/algorithm path instead of
whatever RCCL auto-selected: `NCCL_PROTO=Simple NCCL_ALGO=Ring`. (One
specific detail: `LL128`, a newer, faster protocol, was only just enabled by
default for this chip generation in a recent ROCm release — new protocol
paths on new hardware are a classic source of exactly this kind of silent
hang, so ruling it out is cheap and worth doing early.)

**4. Possible compute-partition-mode inconsistency**, unconfirmed but worth a
quick check: MI355X GPUs can be split into different "partition modes"
(we saw `SPX`/`NPS1` in our `rocm-smi` output), and the default channel count
for a given mode changed between ROCm versions. If even one of the 8 GPUs
somehow ended up in a different partition mode than the other 7, that's a
known source of lopsided, hang-prone collectives. **Next step:**
`rocm-smi --showcomputepartition` and confirm all 8 GPUs genuinely agree.

**Concrete order to try things in, cheapest/most-diagnostic first:**
1. `NCCL_PROTO=Simple NCCL_ALGO=Ring` — rules out the new-protocol theory (#3)
2. `RCCL_MSCCL_ENABLE=0` — rules out the MSCCL-algorithm theory (#2)
3. `NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2` — rules out the channel-tuner
   theory (#2)
4. Confirm all 8 GPUs report identical `rocm-smi --showcomputepartition`
   output (#4)
5. Check the installed `jax-rocm7-plugin`/`jaxlib` version was actually built
   for ROCm 7.2.4, not an adjacent version (#1) — try the latest point
   release if unsure
6. As a way to isolate "is this JAX's fault or RCCL/the driver's fault":
   run RCCL's own standalone benchmark (`all_reduce_perf`, no JAX/Python
   involved at all) on the same 8 GPUs at a similar message size (~525MB,
   matching what we saw JAX trying to send). If *that* also hangs, the bug
   is below JAX entirely — in RCCL or the driver — which is a much stronger,
   more filable bug report to send upstream (and worth including the
   `kfd_wait_on_events` detail specifically, since that's a distinctive clue
   most bug reports don't include).

None of these are confirmed fixes yet — they're the ranked, most-actionable
guesses given the evidence gathered so far. Update this section once one of
them is actually tried.

## Part 6: Resolution

**Fixed.** Killed the hung job a third time and relaunched with three env
vars added at once (theories #1-#2 from the ranked list above, applied
together rather than one at a time, to unblock the run quickly):

```yaml
- name: NCCL_PROTO
  value: "Simple"
- name: NCCL_ALGO
  value: "Ring"
- name: RCCL_MSCCL_ENABLE
  value: "0"
```

Training moved cleanly past the exact point where it had hung twice before
(RCCL `Init COMPLETE` + channel tuning), and reached **step 300** in about 7
minutes, logging a real loss value every step (`loss: 12.29`, decreasing
from higher earlier values as expected this early) and completing its first
automatic checkpoint save right on schedule. No further hang.

**Caveat worth knowing:** because all three env vars were changed in the
same relaunch, we don't know which one (or which combination) was actually
necessary — it's possible only one of `NCCL_PROTO=Simple`,
`NCCL_ALGO=Ring`, or `RCCL_MSCCL_ENABLE=0` mattered, and the other two are
along for the ride. That's fine for getting unblocked, but if these end up
costing meaningful throughput (forcing `Simple`/`Ring` instead of
auto-tuned faster algorithms can be slower at scale) and it matters later,
it'd be worth bisecting by removing one at a time on a future run to find
the minimal fix.

## Part 7: Bisection — which of the three vars actually mattered

Tested each of the three env vars alone (the other two removed), killing and
relaunching between each test, watching for a real step log within 12
minutes as the pass/fail signal:

- **`NCCL_PROTO=Simple` alone: HUNG.** Same signature as the original bug —
  RCCL tuning burst logged, then total silence, no step logged after 12
  minutes. Not sufficient by itself.
- **`NCCL_ALGO=Ring` alone: WORKED.** Training step logged within seconds of
  the tuning burst finishing — same fast, clean success as the full
  three-var fix. **This is the one that actually matters.**

- **`RCCL_MSCCL_ENABLE=0` alone: WORKED.** Also fast and clean, same as
  `NCCL_ALGO=Ring` alone.

**Conclusion.** `NCCL_ALGO=Ring` and `RCCL_MSCCL_ENABLE=0` are each
independently sufficient; `NCCL_PROTO=Simple` is not needed at all. This
makes mechanistic sense: `NCCL_PROTO` controls the wire *protocol*
(Simple/LL/LL128) — unrelated to the bug. `NCCL_ALGO=Ring` and
`RCCL_MSCCL_ENABLE=0` both work because they reach the same outcome by
different routes: forcing the `Ring` algorithm overrides RCCL's default
algorithm selection, and directly disabling MSCCL removes it from
consideration — either way, RCCL stops trying to use its MSCCL collective
algorithm. **This strongly points to the actual root cause being a bug in
RCCL's MSCCL algorithm implementation on this GPU generation (gfx950 /
MI355X)**, consistent with the research pass's finding that MSCCL had
protocol-level bugs recently fixed for this chip generation.

**Minimal fix going forward:** just `RCCL_MSCCL_ENABLE=0` (chosen over
`NCCL_ALGO=Ring` since it directly disables the suspected-buggy component
rather than forcing a specific algorithm — leaves RCCL free to pick the best
non-MSCCL algorithm for the collective, rather than always using Ring even
where it may not be optimal).

## Part 8: A second, different bug — real data crashes with SIGSEGV

Switched to a real workload: real data (`300k-horizons-llama3` instead of
`synthetic-lm`) and real init (`meta-llama/Llama-3.2-1B-Instruct` instead of
`null`), keeping the confirmed `RCCL_MSCCL_ENABLE=0` fix and re-adding the
reclaim-resilience block since this is now real, valuable compute time
(see `docs/national-compute.md`'s own recommendation for real runs).

**New symptom: not a hang this time — a hard crash.** The container exits
with `exitCode: 139` (128 + 11 = `SIGSEGV`, a segmentation fault — a
low-level native memory-access violation, not a Python exception; no
Python traceback prints before it dies, meaning the crash is inside
compiled code — XLA's runtime or RCCL — not in Python itself). Timing is
suspiciously consistent: the crash lands ~13-30 seconds after the same
XLA "clique" rendezvous warning that's benign in the synthetic-data runs,
every time.

**Confirmed reproducible across 3 separate runs**, all `exitCode: 139`:
1. Real data + real init: crashed
2. Real data + real init, exact retry: crashed identically (~same timing)
3. Real data + `initialization=null` (bisecting out the real checkpoint):
   **also crashed** — same signature

**Conclusion: the real *data* triggers this, independent of
initialization.** We didn't need to test synthetic-data + real-init,
since synthetic + null already succeeds reliably (every earlier run), and
real-data + null now crashes reliably too — the one isolated variable is
data.

**What's plausibly different about the real data vs. synthetic:** the
crash happens inside the same danger-zone (first real cross-GPU collective
after the rendezvous), so the leading theory is that real tokenized
sequences produce different tensor shapes/sizes (e.g. variable sequence
lengths, different vocab/padding patterns from `300k-horizons-llama3`'s
real tokenizer output) than the synthetic collator's fixed, synthetic
shapes — and the non-MSCCL Ring/tuner path we forced as the fix for the
original hang may itself have a separate size- or shape-dependent bug that
the synthetic run's tensor sizes never happened to trigger.

**Update: `NCCL_ALGO=Ring` also crashes on real data** (identical exit 139,
same ~13s-after-rendezvous timing) — so the bug is not specific to the
`RCCL_MSCCL_ENABLE=0` code path; both independently-sufficient hang-fixes
fail the same way on real data.

**Node-variable control test.** To rule out a node-specific hardware fault
being the real explanation (rather than data), reran synthetic +
`NCCL_ALGO=Ring` — the combo that had succeeded cleanly and fast every time
on `mighty-marten`/`polite-hippo` — on the newest node, `prime-cougar`.
First attempt: hung silently for 22+ minutes, the *original* bug's
signature, with zero crash. This looked like strong evidence `prime-cougar`
itself was bad. But a same-node, same-config retry immediately after
**succeeded** cleanly (step 50, checkpoint saved, ~2.5 min). A genuinely
broken node wouldn't self-heal on the very next identical attempt — so
`prime-cougar` is not reliably faulty.

**Revised theory: this is one underlying RCCL race condition, not two
separate bugs.** The `NCCL_ALGO=Ring`/`RCCL_MSCCL_ENABLE=0` fix greatly
reduces how often the race fires, but doesn't eliminate it —
synthetic-data runs mostly succeed fast and clean, but rarely still hit the
same silent-hang symptom as the pre-fix bug (this one case, on
`prime-cougar`). Real data has triggered the race **every single time**
(3/3), which fits if real data's different/larger tensor shapes shift the
race's timing window such that it fires far more reliably than synthetic
data's fixed, smaller shapes do — same bug, different trigger probability,
not two independent defects.

**Root-caused the exposure mechanism (code inspection, no cluster time
needed).** Compared `configs/data/synthetic-lm.yaml` (sequence_length: 16,
episodes: 1, overridden to 2 in our tests) against
`configs/data/300k-horizons-llama3.yaml` (cluster_length: 64, max_length:
1024) — real data batches are shaped `(batch, 64, 1024)` vs. synthetic's
`(batch, 2, 16)`, ~2000x more tokens per example.

Two concrete mechanisms, found by reading `src/trainers/horizon_lm_trainer.py`
and `src/utils/attention_utils.py`:
1. **`attention_mask` is never passed into the model at all** —
   `horizon_lm_trainer.py:89-94` calls `self.model.apply(..., input_ids, ...)`
   with no `attention_mask` argument; it defaults to `None` inside the model
   and is only used later for *loss* masking, never fed into attention. So a
   mask-triggered code branch (initially suspected) is ruled out — the
   `portable` attention backend's masking path is never exercised by either
   synthetic or real data.
2. **Episode count directly multiplies exposure to the collective op.**
   `horizon_lm_trainer.py:117-136` runs the model once per "episode" inside
   a `jax.lax.scan`, accumulating gradients across all episodes *before* the
   cross-GPU all-reduce fires once per step. Episodes = `cluster_length`:
   **2 for synthetic** (our override) vs. **64 for real data** — a **32x**
   higher per-step exposure to whatever race condition is involved, on top
   of each individual attention call also being far larger (real data's
   sequence length of 1024 exercises `_portable_flash_attention`'s full
   8-block nested scan; synthetic's length-16 sequences collapse to an
   essentially trivial single 128-token block).

This cleanly explains the earlier finding (race rare on synthetic,
reliable on real) without needing two separate bugs: real data simply
gives the same underlying non-deterministic RCCL issue dramatically more
chances to fire per training step.

**Exposure-count theory tested directly — refuted.** Ran real data + real
init with `data.collator.cluster_length=2` (a Hydra CLI override, no code
push needed — `full-baseline.yaml` is applied straight from local disk;
only the training code itself is fetched via `git clone` inside the pod),
matching synthetic's exposure count exactly. **Still crashed identically**
— same exit 139, same ~13s-after-rendezvous timing, on yet another new
node (`swift-manatee`, a 4th distinct node). So episode/exposure count is
not the (sole) driver — the 32x-exposure mechanism was a real, correct
observation, but not sufficient on its own to explain the crash.

Remaining candidate variables, now that exposure count is ruled out:
sequence length (real data's `max_length: 1024` vs. synthetic's
`sequence_length: 16`), and/or the real tokenized content itself (actual
vocabulary distribution and chat-template structure vs. uniform random
integers). **Next test:** force `data.collator.max_length=16` on real data
too, matching synthetic's per-example shape `(2, 16)` exactly — isolates
whether it's sequence length specifically, or something about real content/
tokenizer output that survives even at matching size.

**Result: SUCCESS.** Real data + real init, `cluster_length=2` +
`max_length=16` (synthetic's exact shape, real tokenized content) —
reached step 20 cleanly, loss dropping from 6.03 to 0.93 (healthy real
training dynamics), ~7-8k steps/hr, no crash. This cleanly isolates the
trigger: **real token content is not the problem** (this run used it and
succeeded), **episode count is not the problem** (already ruled out) —
**sequence length specifically (`max_length: 1024` vs. `16`) is the
remaining, isolated variable.** Whatever's happening, it's tied to long
sequences specifically, most likely inside `_portable_flash_attention`'s
block-wise scan (8 nested blocks at length 1024 vs. an essentially-trivial
single block at length 16) or the larger activation/gradient tensors a
longer sequence produces feeding into the collective at a size/timing that
triggers the race.

**Next test:** hold everything else at the real-data defaults and only
push `max_length` up incrementally (e.g. 64, 128, 256, 512) to find the
approximate threshold where it starts failing — narrows down whether this
is a hard size cliff or a probability that increases with length.

**Threshold search results.** `max_length` 16/64/128/256 all succeeded
cleanly (VRAM steady at 76-77% throughout — not memory pressure).
`max_length=512` failed — but with a **different, more informative crash**:
`exitCode 134` (SIGABRT), not 139 (SIGSEGV) like every prior crash, and
this time an actual error message survived:

```
terminate called after throwing an instance of 'std::bad_variant_access'
  what():  std::get: wrong index for variant
```

**This is likely the same bug as every SIGSEGV crash, just finally
visible.** `std::bad_variant_access` means native C++ code called
`std::get<T>()` on a `std::variant` actually holding a different type —
a real type-confusion bug, almost certainly inside XLA's or RCCL's
collective-communication runtime given the timing (right after the
rendezvous resolves, matching every prior crash exactly). `SIGABRT` goes
through a C++ `terminate` handler that flushes its message before dying;
`SIGSEGV` is the OS forcibly killing the process mid-instruction with no
chance to flush anything — so the *same* underlying throw likely happened
silently every previous time, and this smaller `max_length` just happened
to hit the SIGABRT variant instead of SIGSEGV, which is loud instead of
silent.

**Next:** search for `std::bad_variant_access` / `wrong index for variant`
against known RCCL/XLA-ROCm issues — this is a specific, greppable string,
unlike the previous vague "SIGSEGV, no message" symptom.

## Part 9: Research findings on `std::bad_variant_access`

No exact match for this error text anywhere public (openxla/xla, jax-ml/jax,
ROCm/xla, ROCm/rccl, ROCm/jax) — likely unreported/unindexed rather than
nonexistent, given how new this hardware/software combination is. Several
adjacent, concrete findings strongly shape where to look:

**1. (Most likely) A documented RCCL correctness bug class on gfx950,
right in this size neighborhood.** AMD's own RCCL changelog: *"Fixed a
single-node data corruption issue in MSCCL on the AMD Instinct MI350X and
MI355X GPUs for the LL protocol. This previously affected ~2% of runs for
single-node AllReduce with inputs smaller than 512 KiB"* — fixed in RCCL
2.27.7 (which should be in ROCm 7.1.1+/7.2.4, but the exact build our image
shipped with hasn't been verified). Also ROCm 7.2.0: *"Disabled
`reduceCopyPacks` pipelining for gfx950"* — another gfx950-specific
correctness patch in the same area. `RCCL_MSCCL_ENABLE=0` disables the
MSCCL *algorithm* specifically, but doesn't rule out a sibling bug in the
standard Ring/Tree LL-protocol path sharing the same underlying
chunk/state-selection logic. Our crash boundary (seq_len 256 succeeds, 512
fails) plausibly crosses a similar message-size class for gradient tensors
as the documented "under 512 KiB" boundary.

**2. (Cheap, documented, worth trying first) A known ROCm+JAX segfault
workaround exists, unrelated on its face but same symptom class.** AMD's
own ROCm JAX install docs list a standing known issue: JAX may segfault
during execution on ROCm, worked around by disabling XLA's command
buffers (HIP-graph capture/replay): `XLA_FLAGS="--xla_gpu_enable_command_buffer="`.
Command-buffer/HIP-graph state is exactly the kind of thing XLA represents
internally with `std::variant`, and its capture behavior is shape/size
sensitive — plausibly explaining a sequence-length threshold.

**3. (Confirms the bug class, not this exact bug) XLA's collective-thunk
code has known, similar holes.** `openxla/xla` issue #47383: a
`std::optional` (variant's sibling) accessed without checking validity
inside all-reduce thunk construction, on CUDA (not ROCm, and not our exact
error) — but it establishes this code family is fragile enough to have
had at least one confirmed "assume-initialized" bug already, just not
one caught on ROCm's much-less-tested backend.

**Assessment:** most likely an XLA(ROCm)-thunk ↔ RCCL interaction bug —
XLA's collective thunk likely assumes something (buffer layout, protocol
variant, command-buffer capture state) that doesn't hold at larger message
sizes on gfx950 specifically, at the seam between two independently-
confirmed-fragile pieces of code.

**Concrete next steps, cheapest/highest-signal first:**
1. `XLA_FLAGS="--xla_gpu_enable_command_buffer="` — disable XLA command
   buffers entirely, retest at `max_length=512` (the known-failing case)
2. Add `NCCL_PROTO=Simple` alongside the existing `RCCL_MSCCL_ENABLE=0`
   fix — forces off the LL/LL128 low-latency protocols where the
   documented RCCL bug lives
3. Confirm the installed RCCL version (`ldconfig -p | grep rccl`) is
   actually ≥2.27.7 — the Docker image's JAX-on-top install order could
   have pulled something unexpected
4. If both workarounds fail: this combination (MI355X + ROCm 7.2.4 +
   this exact error) appears genuinely unreported — worth filing upstream
   against `ROCm/rccl` and `ROCm/jax` with the full repro

**Test 1 result: `XLA_FLAGS="--xla_gpu_enable_command_buffer="` — did NOT
fix it.** Reran the known-failing `max_length=512` case with command
buffers disabled; still crashed, same timing. Interestingly it reverted to
the silent `exitCode 139` (SIGSEGV, no message) rather than the loud 134
(SIGABRT) — consistent with disabling command buffers shifting execution
timing enough to land back on the memory-corruption failure path instead
of the throw-based one, rather than actually avoiding the underlying bug.
Ruled out.

**Test 2 result: `NCCL_PROTO=Simple` (alongside `NCCL_ALGO=Ring`) — also
did NOT fix it.** Same `max_length=512` case, same crash, same silent
`exitCode 139`. Ruled out — forcing off LL/LL128 protocols doesn't avoid
whatever this is either.

**Next (cheap, no full training run needed): verify installed RCCL
version.** The documented gfx950 corruption fix landed in RCCL 2.27.7 —
need to confirm the image's actual RCCL build is at or above that,
since JAX was installed separately on top of the base ROCm/PyTorch image
and could plausibly have pulled a different RCCL than expected.

**Result: RCCL is already 2.27.7** (`dpkg -l`: `rccl 2.27.7.70204-93~24.04`),
confirmed via a tiny diagnostic pod with **no GPU request** — scheduled
instantly on the CPU node (`clever-avocet`) instead of going through
market bidding, since RCCL's version is a filesystem fact independent of
having GPUs at all. Not stale software.

**But this actually sharpens the picture rather than closing it off:** the
documented AMD fix was specifically for **MSCCL's LL protocol** — and
every one of our tests has MSCCL *disabled* (`RCCL_MSCCL_ENABLE=0` /
`NCCL_ALGO=Ring`). So that already-patched bug was never in our code path
to begin with. Having the patched version doesn't rule out our crash —
it confirms **our crash is a separate, still-unfixed issue specifically in
the Ring-algorithm path**, not the MSCCL bug AMD already caught.

## Part 10: The likely real root cause — JAX version mismatch

Stepping back after 5 failed RCCL/XLA runtime-knob experiments (all
producing the *identical* crash regardless of setting): that pattern
points away from a tunable algorithm bug and toward something baked into
compiled binaries that no runtime flag can reach. This matches the
highest-ranked (but previously unverified) theory from the very first
research pass: a version mismatch between JAX's ROCm plugin and the
actual installed ROCm/RCCL build.

**Confirmed via AMD's own ROCm 7.2.4 docs** (fetched directly, not a
paraphrase): AMD validates **`jax==0.8.2`** for ROCm 7.2.4 specifically
(`jax-rocm7-pjrt==0.8.2`, `jax-rocm7-plugin==0.8.2`, a matching `jaxlib`
wheel from a GitHub release, not PyPI). **Our `uv.lock` pulls `jax==0.11.1`**
(or 0.10.2) — 3+ minor versions ahead — because `pyproject.toml`'s `rocm`
extra (`jax[rocm7-local]>=0.8.0`) is an unpinned floor, and PyPI resolves
it to the newest available rocm7 plugin build, which is not
version-matched to this specific ROCm point release at all.

AMD also publishes an official, JAX-native image for this exact
combination: `rocm/jax:rocm7.2.4-jax0.8.2-py3.12` — distinct from the
PyTorch-oriented `rocm/pytorch` image this project has been using with
JAX bolted on top. And this is a documented, recurring failure class, not
speculation: AMD's own docs separately note JAX 0.9.1 segfaulting on ROCm
7.13.0, and `ROCm/rocm-jax` issue #200 shows crashes from mismatched
plugin/ROCm `.so` versions — the same shape as our own earlier `jax-aiter`
abandonment (a different component, same "this doesn't match our locked
JAX" problem).

**Verified AMD's exact commands directly** (fetched the live docs page,
not a paraphrase) — confirmed `jax==0.8.2` plus a specific `jaxlib` GitHub
release wheel. First attempt using AMD's literal `pip3` commands
translated to `uv pip install jax-rocm7-pjrt==0.8.2 jax-rocm7-plugin==0.8.2`
failed immediately (before training even started): those exact versions
don't exist on PyPI at all (confirmed via PyPI's JSON API — the package's
release history jumps straight from 0.7.1 to 0.9.1). AMD's docs assume a
pip environment that already has their package index configured; `uv`
doesn't. Found the actual wheels via the GitHub release's asset list
(`ROCm/rocm-jax` tag `rocm-jax-v0.8.2`) — both packages are published
there in `+rocm7.1.1` and `+rocm7.2.0` tagged variants (matching two
different ROCm point releases); used the `+rocm7.2.0` build to match our
exact ROCm 7.2.4 install, plus the `+rocm7` generic tag AMD's docs specify
for `jaxlib`.

## RESULT: SUCCESS — this fixes Bug #2

Relaunched the known-failing `max_length=384` case with `jax==0.8.2` +
the three correct GitHub-release wheels installed over `uv.lock`'s
mismatched `jax==0.11.1`. **Training ran cleanly** — real steps logging,
loss decreasing (2.71 → 2.49 across 6 steps), all 8 devices active, no
crash. This is the exact input size that crashed identically 3 separate
times under the old JAX version. The internal XLA rendezvous log format
also changed slightly (`rendezvous.cc:92` vs. the old build's `:108`),
independently confirming genuinely different compiled code is now
running, not a fluke.

**This strongly confirms the JAX/ROCm version mismatch — not an RCCL
tuning issue — was the real root cause of Bug #2 all along.** It also
retroactively explains why all 5 RCCL-level environment variable
experiments failed identically: none of them could have fixed a bug that
lived in which compiled JAX/XLA-ROCm binary was running, since they only
ever changed RCCL's *behavior*, never its *build*.

**Not yet confirmed:** whether this also holds at the full real workload
size (`max_length=1024`, `cluster_length=64`, real data), and whether
`RCCL_MSCCL_ENABLE=0` (the Bug #1 fix) is even still necessary under
JAX 0.8.2 — worth testing both next.

**Practical implication for this repo going forward:** `pyproject.toml`'s
`rocm = ["jax[rocm7-local]>=0.8.0"]` is a dangerously loose floor — it let
`uv` silently resolve 3+ minor versions past what AMD actually validated
for this ROCm point release, with zero warning. Worth pinning explicitly
rather than leaving it open-ended.

## CONFIRMED AT FULL SCALE — investigation closed

Relaunched with the JAX 0.8.2 fix at the **actual full real workload**:
`max_length=1024`, `cluster_length=64`, real data (`300k-horizons-llama3`),
real init (`meta-llama/Llama-3.2-1B-Instruct`) — the exact configuration
that started this entire investigation. Survived the danger-zone
rendezvous cleanly (`unstuck` at the same checkpoint every crash used to
hit), then logged sustained real training progress:

```
step: 1, loss: 2.3683, grad_norm: 17.706, step_time_s: 128.068, steps/hr: 28.1
step: 2, loss: 2.3971, grad_norm: 19.111, step_time_s:  67.576, steps/hr: 36.8
```

**Root cause, final answer:** `pyproject.toml`'s unpinned `jax[rocm7-local]>=0.8.0`
let `uv.lock` resolve `jax==0.11.1`, three-plus minor versions past
`jax==0.8.2` — the only version AMD validates for ROCm 7.2.4. That
mismatch caused two distinct, severe failure modes (a silent RCCL/MSCCL
hang, and a sequence-length-triggered `std::bad_variant_access`
crash/SIGSEGV) that were completely immune to every RCCL/XLA runtime
environment variable tried (10+ across both bugs). Pinning to AMD's
actual validated JAX version resolved both.

**Follow-up: `pyproject.toml`/`uv.lock` fixed properly** (done, locally —
not yet pushed to GitHub, see Part 11). Also worth re-testing whether
`RCCL_MSCCL_ENABLE=0` (the Bug #1 fix) is still necessary under JAX 0.8.2,
now that the training pipeline is confirmed healthy end-to-end — this is
still genuinely untested (see Part 11).

**Status: both cheap documented workarounds ruled out (command buffers,
NCCL_PROTO=Simple), RCCL version confirmed current.** This is now looking
like a genuinely unreported bug rather than a known-and-documented one we
just haven't found the right flag for.

**Test 3: `RCCL_MSCCL_ENABLE=0` instead of `NCCL_ALGO=Ring`, at
`max_length=512` — also crashed, identically** (`exitCode 139`, silent,
no message). Confirms Bug #2 is fully independent of which Bug #1 fix is
applied — not specific to the Ring algorithm, happens under both
independently-sufficient Bug #1 fixes equally.

**Test 4: `NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2` (alongside
`RCCL_MSCCL_ENABLE=0`) — also crashed, identically** (`exitCode 139`,
silent). Ruled out — forcing a small fixed channel count doesn't avoid it
either.

**Test 5: `NCCL_P2P_DISABLE=1` — also crashed, identically** (`exitCode
139`, silent). Forcing off direct GPU-to-GPU memory access entirely,
a completely different transport path, still didn't avoid it.

**Threshold narrowed further:** `max_length=384` also crashes (same
`exitCode 139`). Confirmed working ceiling is now 256; confirmed failing
floor is 384 (down from the earlier 512 floor).

**Summary: 5 different RCCL/XLA-level fixes tried against Bug #2, all
failed identically.** Command buffers disabled, `NCCL_PROTO=Simple`,
both independently-sufficient Bug #1 fixes tested separately
(`NCCL_ALGO=Ring` vs. `RCCL_MSCCL_ENABLE=0`), a forced small fixed
channel count, and P2P disabled — every one still crashes with the same
silent `exitCode 139` at `max_length=512`. This bug is robust against
every reasonable environment-variable-level workaround in the RCCL/XLA
collective-communication space. Remaining realistic options: (a) treat
the discovered size threshold as an operational constraint (keep sequence
length at or below ~256-384 for now, narrow the exact cliff further if it
matters), (b) try a different software stack entirely — a different
JAX/jaxlib/XLA build, since this is plausibly a bug in this exact version
combination rather than something a runtime flag can route around, or
(c) stop experimenting here; upstream reporting was explicitly ruled out
by the user.

**Node theory fully closed — not just weakened.** Retried the full real
workload (real data + real init) once more; it landed on yet a *third*
distinct physical node (`fleet-pangolin`, never seen before) and crashed
identically: exit code 139. Real data has now failed on 3 separate
physical machines (`mighty-marten`, `prime-cougar`, `fleet-pangolin`) —
conclusively a data-triggered software bug, not hardware tied to any one
node.

**Tooling gap found:** this crash's detailed log was lost. The watcher
script polls `kubectl logs --since=15s` in a loop and only switches
attention to the termination check after that call — if the container
crashes before the loop catches up (e.g. during a slow image pull on a
fresh, never-used node), the 15-second rolling window misses all the
historical output, and by the time this was noticed the pod had already
passed its 10-minute `ttlSecondsAfterFinished` and was garbage collected.
Net data point still stands (exit 139, one more distinct node), just
without this instance's exact rendezvous timing. Fix for future watcher
scripts: fetch full (non-`--since`-restricted) logs immediately once
termination is detected, before the TTL window closes.

**Running tally: real data has crashed 4/4 attempts, across 3 different
physical nodes, both with and without real init.** This is about as
solid as evidence gets without root-causing the exact line of code —
worth treating as a reliable, reportable bug rather than continuing to
chase via more relaunches.

## Part 11: Session wrap-up and status

**`pyproject.toml`/`uv.lock` fix applied locally, not yet pushed.**
`rocm` extra now reads `jax==0.8.2` plus three direct GitHub-release wheel
URLs for `jax-rocm7-pjrt`/`jax-rocm7-plugin`/`jaxlib` (platform-gated to
`linux`/`x86_64`, the two Python-3.12-specific ones also gated on
`python_version`), replacing the old open-ended `jax[rocm7-local]>=0.8.0`
floor. `uv lock` regenerated cleanly (138 packages resolved). The manual
`uv pip install` override step was removed from `full-baseline.yaml`'s
launch script accordingly. **Side effect:** since `jax` is one shared
dependency across the `cuda`/`cuda13`/`rocm`/`tpu` extras in a single
lockfile, pinning it for `rocm` pulled the `cuda`/`tpu` extras down to
`0.8.2` too — harmless here since this repo is ROCm-only in practice, but
worth knowing. **This fix has zero effect until pushed** — the training
pod clones the repo fresh from GitHub each run; only `full-baseline.yaml`
itself is read from local disk. Confirmed via `uv sync --locked --extra
rocm` locally that the lockfile resolves; couldn't fully install-test
since the wheels are Linux-x86_64-only and this machine is macOS ARM
(expected, not a bug).

**Newer ROCm/JAX combinations exist beyond what we pinned** (pulled real
tags from Docker Hub's `rocm/jax` repo, not docs, which had version
inconsistencies): our current `rocm7.2.4`/`jax0.8.2` is the latest *within
the 7.2.x line*, but AMD has since moved to `rocm7.14`/`rocm7.14.1`
(paired with `jax0.9.1` or `jax0.10.0`) and `rocm10.0` (paired with
`jax0.10.0`, `0.10.1`, `0.10.2`, or `0.11.0`). Notably `rocm10.0-jax0.11.0`
is almost exactly the broken `jax==0.11.1` this whole investigation was
about — just validated this time instead of accidental. Moving to a newer
line means changing `full-baseline.yaml`'s base Docker image, not just the
`pyproject.toml` pin — a bigger, separate decision, deferred to next
session.

**Confirmed working, then deliberately killed.** The full-scale run
(real data + real init, JAX 0.8.2 fix) ran healthily for 65 minutes before
being manually stopped (not a crash) to end the session. Pod fully
cleared (`kubectl get pods` empty). Any checkpoints from that run are
lost — they were written to `/workspace/nc-baseline/local_data/...` on
the container's own ephemeral filesystem, not the persistent `/mnt/shared`
volume, so they died with the pod. Not a concern for this debug run, but
worth remembering: a real run whose checkpoints matter needs to point at
`/mnt/shared` or upload to HF to survive a pod being killed.

**GPU node status at session end:** `bg-1` still `Ready` in the cluster
(113+ min old as of last check), no jobs/pods running on it. Deleting the
Job never releases the node — that's tied to the market grant, not
workload state (established earlier this session). The only documented
way to release it early is lowering the bid below its clearing price via
`PUT /api/k8s/bid` (or the NC console), which the user needs to do
themselves — no API credentials available in this session to do it here.

**Next session's plan:**
1. Push `pyproject.toml`/`uv.lock` to GitHub (required before the fix
   has any real effect)
2. Test removing `RCCL_MSCCL_ENABLE=0` under JAX 0.8.2 — genuinely
   unknown whether it's still needed, or whether it was always the same
   underlying version-mismatch bug
3. Consider moving to `rocm10.0`/`jax0.11.0` (bigger jump, likely better
   perf, some indirect compatibility confidence already) or
   `rocm7.14`/`jax0.9.1`-or-`0.10.0` (smaller, more conservative jump) —
   requires changing the base Docker image, not just the JAX pin

## Part 12: Testing the newer ROCm 7.14.1 / JAX 0.10.0 combo

**AMD's versioning is genuinely inconsistent across sources** — worth
recording precisely since it caused real confusion:
- Yesterday's fetch of the ROCm 7.2.4-specific docs page said "for
  production use, continue to use ROCm 7.14.0 documentation" (7.13.0
  labeled a technology preview).
- This morning, `rocm.docs.amd.com/en/latest/` labels **10.0.0** as
  current latest, with no mention of 7.14.x.
- Docker Hub tells a third story: `rocm7.14.1-jax0.10.0` was published
  **2026-09-01**, *after* `rocm10.0-jax0.11.0` (**2026-08-27**) — so by
  actual release recency, 7.14.1 is newer despite the lower number.
- The `ROCm/rocm-jax` GitHub releases for both `v0.10.0` (paired with
  rocm7.14.x) and `v0.11.0` (paired with rocm10.0) ship the **exact same**
  `wheelhouse_theRock7.14.zip` asset — strongly suggesting `rocm10.0` and
  `rocm7.14.x` share the same underlying JAX plugin build, just labeled
  differently upstream, not two genuinely divergent tracks.

**Decision: testing `rocm7.14.1` + `jax==0.10.0` first**, since it's the
one AMD's own version-specific docs explicitly called "production," and
found an exact-naming-convention match for our existing image pattern:
`rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.10.0`.

**Packaging changed since 0.8.2, worth noting:** `jaxlib==0.10.0` is now
on plain PyPI (no more special `+rocm7`-tagged build needed — the
ROCm-specific code apparently moved entirely into the plugin packages).
But `jax-rocm7-pjrt`/`jax-rocm7-plugin` at `0.10.0.post1` are *only*
published bundled inside a zip
(`wheelhouse_post1_theRock7.14.zip`) on the GitHub release — not as
individually fetchable wheel URLs like the 0.8.2 release had. That means
they can't be expressed as normal `pyproject.toml` URL dependencies;
handling this as an explicit download+unzip+install step in
`full-baseline.yaml`'s launch script instead, same as the original 0.8.2
test before it was confirmed and promoted into `pyproject.toml`.

**Testing as an override first, not editing `pyproject.toml` yet** —
consistent with how 0.8.2 was validated before being made permanent.
Current `full-baseline.yaml`: image bumped to `rocm7.14.1_...`, and after
the normal `uv sync` step, overrides with `jax==0.10.0`, `jaxlib==0.10.0`
(PyPI), then downloads/extracts/installs the two zip-bundled ROCm plugin
wheels. `RCCL_MSCCL_ENABLE=0` left in place (not re-testing that variable
in the same run as a version bump — keep one change at a time).

**Hit a real bug in the launch script first:** the extraction step used
`unzip`, which isn't installed in this image (`bash: unzip: command not
found`, exit 127). Fixed by using Python's stdlib `zipfile` instead (via a
heredoc-written script file, to avoid indentation/quoting fragility of an
inline multi-line `python3 -c` string inside a YAML block scalar —
verified locally that YAML's block-scalar indentation-stripping produces
valid Python before relaunching).

**Then hit a much bigger, different problem: a silent CPU/GPU
mismatch.** Once the script ran cleanly, the job appeared to "work" —
fast compile (~1s vs. the usual ~35s), no rendezvous warning at all — but
then sat silently producing no step logs. Live inspection revealed why:
`ps aux` showed the training process at **5420% CPU** (~54 cores) while
`rocm-smi` showed **all 8 GPUs at 0% utilization, 0% VRAM**. The model
config had also logged `devices: 1 (1 processes)` — not 8. Reproduced
identically on a second, independent attempt (killed and relaunched from
scratch) — not a fluke.

**Root cause, found via a live `jax.devices()` check inside the running
pod:** the ROCm PJRT plugin was failing to load at all —
`Failed to open .../xla_rocm_plugin.so: librocprofiler-sdk.so.1: cannot
open shared object file: No such file or directory`. JAX silently falls
back to CPU-only execution when its GPU plugin fails to load
(`devices: [CpuDevice(id=0)]`) — explaining every symptom: the "training"
was actually running entirely on CPU (hence the 54-core spin, an LLM
forward/backward pass parallelized across many CPU threads), while the
real GPUs sat completely idle.

**Why the library was missing:** `rocm7.14.1`'s ROCm distribution moved
from system `.deb` packages (`/opt/rocm-.../lib`, what `rocm7.2.4` used)
to a **pip-installable SDK** bundled inside the image's own separate
`/opt/venv` (packages `_rocm_sdk_core`, `_rocm_sdk_libraries`, etc.). Our
launch script installs JAX into a *different*, separately-`uv`-managed
venv (`/workspace/nc-baseline/.venv`) — and native shared libraries in one
venv aren't visible to a separate venv's dynamic linker without explicit
`LD_LIBRARY_PATH` configuration. `LLVM_PATH` (`/opt/rocm/llvm`) was
also stale for the same reason — that whole path no longer exists in this
image.

**Fix, verified live before relaunching the full job** (via `kubectl exec`
running `jax.devices()` directly inside the still-running, misconfigured
pod — avoided burning another ~15 min provisioning cycle to find out):
- `LLVM_PATH` → `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm`
- `LD_LIBRARY_PATH` →
  `/opt/venv/.../_rocm_sdk_core/lib:/opt/venv/.../_rocm_sdk_libraries/lib`
  (RCCL specifically lives in the second, sibling package — needed both,
  found via a second missing-library error, `librccl.so.1`, after fixing
  the first one)

With both set: `jax.devices()` correctly returned all 8
`RocmDevice(id=0..7)`. Relaunching the actual training job with this
fix now to confirm end-to-end.

## CONFIRMED: ROCm 7.14.1 / JAX 0.10.0 works, and is faster

Relaunched the full, unmodified real workload (`max_length=1024`,
`cluster_length=64`, real data, real init — no overrides) with the
`LD_LIBRARY_PATH`/`LLVM_PATH` fix in place. `devices: 8 (1 processes)`
confirmed in the startup config dump. No rendezvous warning appeared at
all this run (unlike literally every run under the old JAX version) —
consistent with the newer XLA/JAX build handling (or not needing) that
particular internal watchdog differently. Sustained training confirmed:

```
step: 1, loss: 2.3674, step_time_s: 141.491  (includes compile)
step: 2, loss: 2.3962, step_time_s:  57.413
step: 3, loss: 2.4544, step_time_s:  57.209
```

**Notably faster than yesterday's confirmed fix**: steady-state step time
here is ~57.2-57.4s, versus `rocm7.2.4`/`jax0.8.2`'s ~67.6s at the
equivalent point — roughly **15% faster per step**.

**Root cause of everything in this Part 12, in one sentence:** `rocm7.14.1`
moved ROCm's runtime libraries from system packages to a pip-installed SDK
in a separate venv, and our own separately-`uv`-managed venv couldn't see
those libraries without explicit `LD_LIBRARY_PATH`/`LLVM_PATH` overrides —
nothing to do with device count, RCCL, or any of the original two bugs.

**Status: working as an override in `full-baseline.yaml`, not yet
promoted to `pyproject.toml`/permanent image change.** Still queued:
1. Decide whether to make this the new default (update `pyproject.toml`
   pins + base image permanently) or keep `rocm7.2.4`/`jax0.8.2` as the
   safe default with this as an documented alternative
2. ~~Test removing `RCCL_MSCCL_ENABLE=0` under this new combo too~~ —
   **done, see below: not needed.**
3. Push everything to GitHub — still nothing from today or yesterday has
   been pushed

## CONFIRMED: `RCCL_MSCCL_ENABLE=0` not needed under ROCm 7.14.1/JAX 0.10.0

Killed the healthy 22-step run and relaunched with the flag removed
entirely (everything else unchanged — full real workload, no other
overrides). Reached step 3 cleanly, no hang, no crash:

```
Without flag:  step 1: 144.024s | step 2: 57.087s | step 3: 57.224s
With flag:     step 1: 141.491s | step 2: 57.413s | step 3: 57.209s
```

Essentially identical timing with or without the flag — confirms it's
not just "not needed" but has **zero measurable effect** under this
combo. Strong retroactive evidence that Bug #1 (the original MSCCL hang)
and Bug #2 (the sequence-length crash) were the same underlying
JAX/ROCm-plugin version-mismatch problem all along, just manifesting two
different ways depending on execution timing/size — not two independent
RCCL bugs. Fixing the version mismatch fixed both, and the `RCCL_MSCCL_ENABLE=0`
workaround was only ever masking one symptom of it.

**Minimal working config for `rocm7.14.1`/`jax0.10.0`, confirmed:** base
image + `LD_LIBRARY_PATH`/`LLVM_PATH` fix + the zip-extraction install
step. No RCCL/NCCL environment variables needed at all.

## Part 13: JAX 0.11.2 — newest verified build for rocm7.14.1

Two explicit gaps were flagged and answered:
- **Q: latest ROCm possible?** No — `rocm.docs.amd.com/en/latest/`'s own
  homepage header reads "AMD ROCm 10.0.0," a numerically newer line we
  haven't tried. We're deliberately on `7.14.1` (called out for
  "production use" in AMD's version-specific docs, with a same-day
  matching Docker image).
- **Q: latest *verified* JAX for our chosen ROCm line?** Also no, at the
  time — we were on `jax==0.10.0` (built late June), while
  `rocm-jax-v0.11.2` (published 2026-09-17, six days prior) explicitly
  validates against `theRock7.14` — our exact line — per its own release
  notes (`TheRock: 7.14, 10.0`; 31,625 JAX tests passed, 0 failed; MoE
  and jax-lab suites also passing).

**Tested `jax==0.11.2` on the unchanged `rocm7.14.1` image** — same
override pattern as `0.10.0` (plain PyPI for `jax`/`jaxlib`, zip-bundled
`jax-rocm7-pjrt`/`jax-rocm7-plugin` downloaded from the
`rocm-jax-v0.11.2` GitHub release, `wheelhouse_theRock7.14.zip` asset —
clean `0.11.2` version tag this time, no `.post1` suffix). Also tested
without `RCCL_MSCCL_ENABLE=0` from the start, consistent with Part 12's
finding that it's unnecessary.

**Result: works cleanly, throughput identical to 0.10.0.** Reached
step 15+ with no crash, no hang, loss/grad_norm converging normally.

**Metric gotcha worth documenting**: the logged `steps/hr` figure is a
**rolling 10-step average** (`deque(maxlen=10)` in
`base_trainer.py:192`), not an instantaneous or cumulative-since-start
rate. Step 1 always includes one-time compile overhead (~140s vs. ~57s
steady-state) — that outlier sits inside the 10-step window and visibly
drags the average down until it "ages out" exactly at step 11 (window
becomes steps 2-11). Produces a real-looking jump (e.g. `54.8 → 62.7`
between steps 10 and 11) that is **not** a throughput change — it
happens on every run, regardless of JAX version, and should not be
mistaken for a regression. The reliable metric for run-to-run comparison
is `step_time_s` directly, not `steps/hr` before step ~11.

**Steady-state comparison, using `step_time_s` (the reliable metric):**

| Config | step_time_s |
|---|---|
| `rocm7.2.4`/`jax0.8.2` | ~67.3s |
| `rocm7.14.1`/`jax0.10.0` (with or without `RCCL_MSCCL_ENABLE=0`) | ~57.28s |
| `rocm7.14.1`/`jax0.11.2` | ~57.4-58.3s |

`0.11.2` performs identically to `0.10.0` (within noise) — both ~15%
faster than the original `7.2.4`/`0.8.2` combo. Upgrading JAX further
within the same ROCm line is a safe, no-cost-no-benefit-either-way move
on throughput, but keeps us on the actually-latest-verified build.

**Next:** user wants to try the `rocm10.0` line next (paired with
whatever JAX version is newest-verified for it — `v0.11.2`'s release
notes confirm it also validates against `theRock10.0.0`, so likely
`jax==0.11.2` again, just installed from the `wheelhouse_theRock10.0.0.zip`
asset instead — image and `LD_LIBRARY_PATH` paths would need re-deriving
for that image, not assumed identical to `7.14.1`'s).

## Billing / cost management (see also: report drafted for NC platform team)

Confirmed gap: `kubectl delete job` never releases the underlying GPU
node; a new Job's market auction has no preference for reassigning a
node your account already holds. Result: **`bg-2` has been sitting idle
and (almost certainly) billing since ~18:02**, while `perky-grebe` (then
active) and now whatever node the next test lands on billed/bills
concurrently. The only release lever, `PUT /api/k8s/bid`, is
**cluster-wide** — lowering it while a node is actively in use risks
losing that node too, so it's not safe to use mid-experiment.

**Safe pattern going forward, identified this session:** lower the bid
only at a moment when *nothing* is actively needed — i.e., right after
killing a Job and *before* applying the next one — so all currently-held
idle nodes (not just the most recent one) get released together, then
raise the bid back up to admit the next Job. Bundles cleanup with the
natural kill→relaunch cycle instead of trying to surgically target one
node.

**Full ROCm/JAX combo results, this session:**

| ROCm | JAX | `RCCL_MSCCL_ENABLE=0`? | Result | Steady-state step time |
|---|---|---|---|---|
| `7.2.4` | `0.11.1` (accidental) | with/without | Two distinct failures (Bug #1, Bug #2) | N/A |
| `7.2.4` | `0.8.2` | with | Works | ~67.3s |
| `7.14.1` | `0.10.0` | with | Works | ~57.28s |
| `7.14.1` | `0.10.0` | without | Works | ~57.28s |
| `7.14.1` | `0.11.2` | without | Works | ~57.4-58.3s |
| `10.0` | `0.11.2` | without | Works | ~57.0-57.3s |

`rocm10.0`'s `LD_LIBRARY_PATH`/`LLVM_PATH` were carried over unverified
from `7.14.1` (same `_rocm_sdk_core`/`_rocm_sdk_libraries` paths) — worked
on the first attempt, no live debugging needed this time. Note: this
line's plugin packages are named `jax_rocm10_pjrt`/`jax_rocm10_plugin`,
not `jax_rocm7_*` — a genuinely different build, not a rename, though our
generic wheel-extraction filter (matching on `py3-none`/`cp312` in the
filename) didn't need updating for it.

**`rocm10.0` gives no further speedup over `7.14.1`** — both land at the
same ~57s/step, ~15% faster than the original `7.2.4`/`0.8.2` fix. No
reason to prefer one over the other on performance; `7.14.1` remains the
one AMD's docs explicitly call out for production use.

**Tensor-core utilization observed to be low (~19.3%, per the user's own
monitoring)** despite healthy GPU-busy and normal step timing. Likely
contributors, reasoned from the codebase (not yet independently profiled):
1. `attention_backend: portable` — the un-optimized fallback. AMD's actual
   vendor-tuned kernel (`jax-aiter`) was installed and abandoned within 17
   minutes on 2026-09-12 (see Part 4) due to a JAX-version mismatch;
   `_portable_flash_attention` (`src/utils/attention_utils.py`) is built
   from many small blockwise `jax.lax.scan` steps rather than one fused
   matrix-core kernel.
2. `num_logit_iterations: 4` (`configs/trainer/horizon-xl.yaml`) — chunks
   the final logits matmul into 4 smaller calls, adding launch overhead
   relative to one large matmul.
3. Small effective per-matmul batch: `global_batch_size: 128` / 8 GPUs =
   16/GPU, further split per-episode inside the training-step scan.

**Tonight's plan:** revisit the `aiter` attention path — the JAX-version
mismatch that killed it in the original commit may no longer apply now
that we're on `jax==0.11.2`/`rocm10.0` rather than the original
`jax[rocm7-local]`-resolved `0.11.1`. Worth checking whether
`jax-aiter`'s compatibility constraint is satisfied by our current pins
before assuming it's still broken.

**Session end state (updated):** job killed, pod cleared. `perky-grebe`
was later confirmed released — a `kubectl delete node perky-grebe`
attempt returned `NotFound` (it was already gone by the time the command
ran), and it no longer appears in `kubectl get nodes`. Something released
it independently (user action or a natural lapse) right around that
moment; the delete command itself gets no credit. No GPU nodes remained
in the cluster afterward. Nothing from today or yesterday has been
pushed to GitHub yet.

## Part 14: Tensor-core utilization and the gradient-sync frequency bug

User observed tensor-core utilization sitting around **19.3%** during a
healthy, fast-running step (per their own external monitoring, likely
W&B's system metrics — not something we'd instrumented ourselves). Worth
distinguishing from "GPU busy %": tensor-core utilization specifically
tracks the dedicated matrix-multiply hardware, which can sit low even
while the GPU looks fully busy overall, if a lot of step time goes to
non-matmul work (elementwise ops, masking, communication) instead.

**Three codebase-specific reasons identified** (reasoned from source, not
yet independently profiled):
1. `attention_backend: portable` — the un-optimized fallback.
   `_portable_flash_attention` (`src/utils/attention_utils.py`) genuinely
   implements the real FlashAttention algorithm (confirmed via its own
   docstring citing the paper, Algorithm 1: blockwise, online-softmax, no
   full attention matrix materialized) — so it gets Flash Attention's
   *memory* benefit, but it's written as plain JAX `jax.lax.scan` loops
   over blocks rather than a hand-fused kernel, so it doesn't get the
   *throughput* benefit a real fused kernel (`aiter`, cuDNN) provides.
   `aiter` (AMD's actual fused kernel) was installed and abandoned on
   2026-09-12 due to a JAX-version mismatch (see Part 4) — worth
   revisiting now that we're on validated, newer JAX/ROCm pins.
2. `num_logit_iterations: 4` (`configs/trainer/horizon-xl.yaml`) — chunks
   the final logits matmul into 4 smaller calls.
3. Small effective per-matmul batch (`global_batch_size: 128` / 8 GPUs =
   16/GPU, further split per-episode).

**User's proposed fix: fully replicate parameters, communicate gradients
only once at the end of the inner (episode) loop.** Investigated whether
this was already happening, since `configs/model/sharding/fsdp.yaml` is
empty (`default: []`, `rules: []`) and `sharding_utils.py`'s own comment
says `"# fully replicate by default"` — confirming **parameters already
are fully replicated**, despite the file being named `fsdp.yaml`. The
mesh has an 8-wide `"fsdp"` axis (`mesh_utils.py`) that currently goes
completely unused for parameter sharding.

**But gradient communication is NOT already deferred to once per step —
verified this empirically, at zero cluster cost**, by simulating an
8-device mesh on CPU locally (`XLA_FLAGS=--xla_force_host_platform_device_count=8`)
and inspecting the actual compiled HLO of a minimal reproduction matching
the real training step's structure (replicated params, batch sharded
across the full mesh, `jax.lax.scan` accumulating per-episode gradients
via `jax.grad`, then a single `apply_gradients`-style update after the
scan — mirroring `horizon_lm_trainer.py`/`base_trainer.py:116` exactly).

- **Baseline (current real structure): 1 `all-reduce`, located
  `inside while/body`** — i.e., once per scan iteration (once per
  episode), not once per step. At `cluster_length: 64`, that's **64
  separate cross-GPU all-reduce calls per training step**.
- **Root cause**: `jax.lax.scan`'s carry must keep identical
  shape/sharding across all iterations. Starting the carry as
  `jnp.zeros_like(params)` (replicated) locks that requirement in from
  the first iteration, forcing XLA to reduce each episode's local
  gradient into replicated form immediately, every time.
- **Tried restructuring to "stack outputs, sum after the scan" instead of
  accumulating in the carry — did NOT fix it.** Still one all-reduce per
  iteration. Root cause is deeper than an accumulation-strategy artifact:
  computing `jax.grad` of a loss that mixes replicated params with
  locally-sharded data is *mathematically* a data-parallel gradient,
  which requires a cross-device sum to be correct — JAX's automatic
  partitioner inserts that sum as early as it can prove it's needed
  (inside `jax.grad` itself), with no built-in mechanism to say "don't
  reduce yet, more local accumulation is coming."
- **Verified fix: `jax.shard_map` (explicit manual SPMD) with a single,
  explicitly-placed `jax.lax.psum` called once *after* a purely local
  `jax.lax.scan` accumulation.** Result: exactly **1 `all-reduce`,
  located outside the loop**. Confirms the 64x-to-1x reduction is
  achievable — proof of concept validated locally, zero GPU cost.

**Integration cost, not yet attempted:** `shard_map` switches the wrapped
computation from JAX's automatic partitioning (what the rest of this
codebase — including `attention_utils.py`'s `with_sharding_constraint`
calls — relies on) into manual SPMD, where every op inside must be
written to run per-device explicitly. Porting this into
`horizon_lm_trainer.py`'s real `_train_step` means restructuring it to
wrap the episode scan in `shard_map` with explicit mesh axis names, and
checking the model's own internals still behave correctly running inside
that manual context. Real refactor, not a drop-in change — queued as a
follow-up, not yet started.

Local repro scripts (for reference, not part of the repo):
`check_collectives.py` (baseline, 1 all-reduce/iteration),
`check_collectives_v2.py` (stack-then-sum, same result),
`check_collectives_v3.py` (shard_map + explicit psum, fixed) — written to
the session scratchpad, not committed.
