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

## Preemption / reclaims are normal — design for them

A node can be reclaimed if the market clears above your ceiling, or
immediately if you lower your own price or run out of balance.

 On reclaim:
the whole job (all pods in the gang) is deleted together and requeued as a
fresh request — this is *not* a failure, but by default it **does** count
against the Job's `backoffLimit` and can exhaust it on a long run purely from
market churn.

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
limit price below what the gang started at voids the protection window but does not always mean it will be taken back within the 1hr window.

### The mechanism: every workload is pinned to its own grant

Each granted node carries a label

```
marketplace.nationalcompute.com/request: <uuid of the grant that created it>
```

and on admission Kueue injects that same uuid into the pod's nodeSelector.
So a pod can only ever land on a node belonging to *its own* grant.

**Nodes are reused sometimes.** If a node has just completed a job and is within the protected window and has had some idle time, a new job will be allocated to it.

**Double-billing if not careful.** If a job is deleted, we might expect the node to be idle but if not enough time is given, the cluster allocates a new node entirely. This results in being billed for two nodes and the new node sticking around for atleast an hour.
