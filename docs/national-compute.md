# National Compute: how job launches work here

Notes on the National Compute (NC) Kubernetes platform, based on their docs
(Burst capacity, Kubernetes clusters) and how it maps onto this repo's job
files (`train-smoke-job.yaml`, `full-baseline.yaml`).

## The core model: a market, not a scheduler

There's no "reserve N GPUs" call. Instead:

- Your **Kubernetes jobs are the demand**. `kubectl apply` a Job/JobSet/MPIJob
  requesting GPUs, and Kueue (installed in-cluster) turns it into one request
  for the whole gang of nodes, priced together.
- Your **limit price** (`PUT /api/k8s/bid`, or a per-job
  `nationalcompute.com/limit-price` label) is the only thing you declare
  separately. It's the max $/GPU-hour you'll pay.
- Every ~10s tick, an auction runs. If your price clears, nodes join your
  cluster and every pod in the gang starts at once. If the market clears
  above your ceiling, nodes are reclaimed and the whole job requeues.
- **No limit price set = $0 bid = nothing ever gets granted.** This is the
  easiest way to have a job sit pending forever with no error.
- You're billed what the auction assesses (second-price/VCG-style), which is
  usually below your ceiling, never above it. Bid your true max once —
  there's no benefit to lowballing or ratcheting up slowly.

## GPU requests must be whole nodes

Capacity is granted in whole nodes only, never split. A pod's GPU request
must be a multiple of `gpus_per_node` (8 on the MI355X island this repo
targets), or `kubectl apply` is refused outright at admission
(`JobSizeTooSmall`).

`train-smoke-job.yaml` requests `amd.com/gpu: 8` — that's exactly one node,
which is why it's a valid request as-is.

## Launch sequence

1. **Connect once**: fetch a kubeconfig with an org API token (`NC_TOKEN`),
   or run NC's `setup.sh`. First `kubectl` call prints a sign-in link a human
   has to click; it self-refreshes after that until ~a day of inactivity.
2. **Set a limit price** before (or instead of relying on) applying any job:
   `PUT /api/k8s/bid?cluster=<cluster>` with `max_price_per_gpu_hour`. Check
   the price feed / price-to-win ladder first to pick a sane number.
3. **Check credit balance**: apply is refused (`BalanceTooLow`) unless your
   org balance covers `2 hours x limit_price x total_GPUs` for the job.
4. **`kubectl apply -f <job>.yaml`**. No queue name, no `suspend: true`
   needed — Kueue handles suspension and injects the GPU-node toleration on
   admission automatically.
5. **Watch it**:
   - `kubectl get workloads` — Kueue's queue entry, admitted or not
   - `kubectl describe job <name>` — market verdict as Events, then one
     `NodeGranted` event per node
   - `kubectl describe provisioningrequest` — the live `Provisioned` condition
   - Pods don't exist until the *entire* gang is granted (all-or-nothing, no
     partial "1 of 2 nodes" state).

## Why a job can sit pending

Three distinct reasons, only visible via the console's Workloads page or
queue events — the raw reason string doesn't distinguish them:

- **`BidTooLow`** — your limit price can't clear the market. Raise it.
- **`capacity unavailable`** — genuine site-wide shortage. Raising your price
  does nothing; it starts when supply returns. Cost is $0 while waiting.
- **`BalanceTooLow`** (re-checked every tick, not just at apply) — top up and
  it clears on the next tick.

## Preemption / reclaims are normal — design for them

A node can be reclaimed if the market clears above your ceiling, or
immediately if you lower your own price or run out of balance. On reclaim:
the whole job (all pods in the gang) is deleted together and requeued as a
fresh request — this is *not* a failure, but by default it **does** count
against the Job's `backoffLimit` and can exhaust it on a long run purely from
market churn.

Mitigation (missing from `train-smoke-job.yaml` today, worth adding for any
real training run, not just the smoke test):

```yaml
spec:
  podReplacementPolicy: Failed
  podFailurePolicy:
    rules:
      - action: Ignore
        onPodConditions:
          - type: DisruptionTarget
            status: "True"
      - action: Ignore
        onExitCodes:
          operator: In
          values: [143]   # SIGTERM exit code
  template:
    spec:
      restartPolicy: Never
      terminationGracePeriodSeconds: 60
```

Other-tenant preemptions get 1 minute's warning (`SIGTERM`, then a
`preempt-at` deadline annotation) to checkpoint. Preemptions caused by your
own account (price cut, withdrawn bid, empty balance) are immediate, no
warning. Either way: checkpoint often, treat node-local disk as ephemeral,
keep real state on the shared volume (`/mnt/shared`).

## Minimum-duration protection

A freshly granted node is protected from competing bids for its first
2 hours (Kubernetes) / 1 hour (VM). The first hour is also a **minimum
billed hold** — the node stays and bills even if your job finishes early, so
short smoke-test jobs still cost a full hour of node time. Lowering your
limit price below what the gang started at voids the protection window.

## Relaunching: a job delete loses your node, and waiting does NOT get it back

**Deleting a job hands its node back to the market. Your next apply gets a
*new* grant — a different node with its own minimum billed hold — while the
node you are still paying for sits idle in the cluster, healthy and
useless.** Tested: waiting for full reconciliation before re-applying does
*not* recover it.

Measured on 2026-09-27, cost roughly two node-hours for one experiment:

```
01:49:29  NodeGranted bg-1; pod starts, image pulled in 1m37s
02:08:38  kubectl delete job          (run 1 hung; relaunch with a config change)
02:08:39  kubectl apply               <-- immediate re-apply
02:09:11  NodeGranted id-4            <-- a DIFFERENT node, provisioning from scratch

          ... delete again, wait for FULL reconciliation:
          0 jobs, 0 workloads, 0 pods, and bg-1 still Ready with
          8 allocatable GPUs, schedulable, untainted beyond the usual
          amd.com/gpu=present:NoSchedule, no deletionTimestamp ...

02:13:58  kubectl apply               <-- after ~5 min of waiting
02:14:24  NodeGranted id-4 AGAIN      <-- still not bg-1
```

### The mechanism: every workload is pinned to its own grant

Each granted node carries a label

```
marketplace.nationalcompute.com/request: <uuid of the grant that created it>
```

and on admission Kueue injects that same uuid into the pod's nodeSelector.
So a pod can only ever land on a node belonging to *its own* grant.

**Nodes are reused — the marketplace relabels them.** The `id-4` grants
above did not always mean a new machine: at 02:32 the market answered a new
workload by reassigning the existing `bg-1` and rewriting its request label
to the new uuid. An earlier claim in this file that a released node can
never be recovered was wrong; it is recovered, just under a new identity.

**What you cannot do is choose which node.** A `nodeSelector` or `nodeName`
of your own ANDs with the market's request-uuid selector, and if the market
picks a different node than you guessed, the pod is unschedulable forever:

```
0/6 nodes are available: 6 node(s) didn't match Pod's node affinity/selector
```

That is exactly what happened at 02:32 — pinned to `bg-2`, market assigned
`bg-1`, deadlock. **Never pin a node in a job manifest on this platform.**
Submit unpinned and let the market place it; two unpinned jobs requesting
8 GPUs each will take one node apiece.

Why the delete still costs you: the minimum-hold rule guarantees a granted
node *stays and bills* for its first hour, but it does not reserve it for
your next job. The hold is a billing floor, not a lease you can re-enter,
and between delete and re-apply the market is free to answer your next
request with a brand-new machine instead.

**Practical consequence — treat a job delete as expensive.** A job's pod
template is immutable, so any config change forces delete + re-apply, and
each one risks another node-hour. Therefore:

- Batch every change you want to test into a single relaunch. Never
  delete to test one env var.
- Prefer experiments you can drive from *inside* the running container
  (`kubectl exec`) over anything requiring a new pod.
- Bake switchable behaviour into the job script — read experiment knobs
  from a ConfigMap or a file on the shared volume that the startup script
  polls — so a new experiment does not need a new pod at all.
- Before deleting, check the console for what the node actually costs and
  how much of its hold remains; if the hold is nearly expired the delete
  is cheap, if it just started it is not.

Diagnosing a job that will not start (the console shows this as "waiting
for market capacity", which conflates two different states):

```bash
kubectl get workload <name> -o jsonpath='{.status.admissionChecks}'
```

- `"waiting for the market"` — no node granted yet: genuine shortage,
  bid too low, or balance too low.
- `"0 of N granted nodes ready"` — a node *has* been granted and is
  booting. Nothing is wrong; it just has not joined yet. The console may
  still render this as "waiting for capacity."

## Base Load Capacity (not used by our job files today)

A separate, non-market product: a fixed block of nodes at a fixed rate for a
fixed term, bought via the console, no bidding/preemption. Jobs don't name
the block — the scheduler just fills it first before any job bids on the
market for the remainder. Only relevant if we buy a reserved block later;
`train-smoke-job.yaml` / `full-baseline.yaml` run purely on the burst market.

## Open questions / not yet in this repo

- No script here sets the limit price (`PUT /api/k8s/bid`) — currently a
  manual `curl` step outside version control.
- `train-smoke-job.yaml` doesn't set a per-job `nationalcompute.com/limit-price`
  label, so it always uses whatever the cluster's bid is set to.
- No `podReplacementPolicy`/`podFailurePolicy`/`terminationGracePeriodSeconds`
  reclaim-resilience block yet (see above) — probably fine for a 3-step
  smoke test, worth adding before any long real training run.
