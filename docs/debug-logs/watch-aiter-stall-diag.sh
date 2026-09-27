#!/bin/bash
# Diagnostic re-run of the aiter step-18 stall. Same training config as the
# stalling run -- no fix attempts, no RCCL_MSCCL_ENABLE=0 -- plus the
# instrumentation block in full-baseline.yaml.
#
# Waits for a GPU node to join the cluster (there is none as of launch), then
# applies the job and streams logs until the job ends or you Ctrl-C.
set -uo pipefail

REPO="/Users/nivedithaiyer/Dev/adam/nc-baseline"
LOGFILE="$REPO/docs/debug-logs/full-baseline-aiter-stall-diag.log"
: > "$LOGFILE"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGFILE"; }

# Kueue owns the waiting, not this script. The cluster's admission webhook
# suspends every job and holds it in the `market` ClusterQueue until the
# marketplace grants nodes, so there is nothing to poll for and no reason to
# apply only once a node exists -- submitting early is how you get in the
# queue. This script therefore never applies over an existing job: a delete
# and re-apply would throw away a queued (or running) workload.
if kubectl get job full-baseline >/dev/null 2>&1; then
  log "Job full-baseline already exists -- attaching, not re-applying."
else
  log "Applying full-baseline.yaml (diagnostic instrumentation enabled)..."
  kubectl apply -f "$REPO/full-baseline.yaml" 2>&1 | tee -a "$LOGFILE"
fi

# Surface the market verdict while we wait; "waiting for the market" with no
# limit price set means a $0 bid, which is never granted (docs/national-compute.md).
kubectl get workloads 2>&1 | tee -a "$LOGFILE"
kubectl get events --field-selector involvedObject.name=full-baseline \
  -o custom-columns='REASON:.reason,MSG:.message' 2>&1 | tail -5 | tee -a "$LOGFILE"

POD=""
for _ in $(seq 1 180); do
  POD=$(kubectl get pods -l job-name=full-baseline \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -n "$POD" ] && break
  sleep 10
done

if [ -z "$POD" ]; then
  log "No pod appeared after ~30 minutes."
  kubectl describe job full-baseline >> "$LOGFILE" 2>&1
  exit 1
fi

# nodeName isn't populated the instant the pod object appears, so re-read it
# until it is rather than logging an empty node.
NODE=""
for _ in $(seq 1 30); do
  NODE=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  [ -n "$NODE" ] && break
  sleep 2
done
log "Pod: $POD on node: ${NODE:-<unassigned>}"

# Wait for the container to actually start before attaching. `kubectl logs -f`
# does NOT block while a container is still ContainerCreating -- it returns
# BadRequest straight away, so attaching in a retry loop just fills the log
# with one error line per retry for the whole (long) image pull.
log "Waiting for the container to start (image pull can take a while)..."
for _ in $(seq 1 360); do  # up to ~1h at 10s
  CSTATE=$(kubectl get pod "$POD" \
    -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null)
  echo "$CSTATE" | grep -q '"waiting"' || break
  sleep 10
done
log "Container started (or pod terminal): ${CSTATE:-unknown}"

# `kubectl logs -f` streams without gaps. The older watch-*.sh scripts polled
# with `--since=30s` every 10s, which is why their logs contain every step
# line two or three times -- don't copy that pattern.
log "--- Streaming logs (reconnects if the stream drops; Ctrl-C to detach) ---"
while true; do
  STATE=$(kubectl get job full-baseline \
    -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  if echo "$STATE" | grep -qE 'Complete|Failed'; then
    log "Job reached terminal state: $STATE"
    break
  fi

  kubectl logs -f "$POD" --timestamps >> "$LOGFILE" 2>&1

  PHASE=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
    log "*** pod reached terminal phase: $PHASE ***"
    kubectl describe pod "$POD" >> "$LOGFILE" 2>&1
    break
  fi
  sleep 5
done

log "--- Final job status ---"
kubectl get job full-baseline -o wide >> "$LOGFILE" 2>&1
