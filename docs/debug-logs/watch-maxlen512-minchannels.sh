#!/bin/bash
set -uo pipefail
LOGFILE="/Users/nivedithaiyer/Dev/adam/nc-baseline/docs/debug-logs/full-baseline-maxlen512-minchannels.log"
VRAMLOG="/Users/nivedithaiyer/Dev/adam/nc-baseline/docs/debug-logs/full-baseline-maxlen512-minchannels-vram.log"
: > "$LOGFILE"
: > "$VRAMLOG"

echo "$(date -u +%FT%TZ) Waiting for pod (max_length=512 (XLA_FLAGS test), cluster_length=2, watching VRAM too)..." | tee -a "$LOGFILE"

POD=""
for i in $(seq 1 180); do
  POD=$(kubectl get pods -l job-name=full-baseline -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$POD" ]; then
    break
  fi
  sleep 10
done

if [ -z "$POD" ]; then
  echo "$(date -u +%FT%TZ) No pod appeared after ~30 minutes." | tee -a "$LOGFILE"
  kubectl describe job full-baseline >> "$LOGFILE" 2>&1
  exit 1
fi

NODE=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
echo "$(date -u +%FT%TZ) Pod found: $POD on node: $NODE" | tee -a "$LOGFILE"

for i in $(seq 1 120); do
  PHASE=$(kubectl get pod "$POD" -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null)
  echo "$(date -u +%FT%TZ) container state: $PHASE" >> "$LOGFILE"
  if echo "$PHASE" | grep -qE 'running|terminated'; then
    break
  fi
  sleep 10
done

echo "$(date -u +%FT%TZ) --- Streaming logs + VRAM, watching for step-log (success), crash, or hang ---" | tee -a "$LOGFILE"

RESULT="unknown"
for i in $(seq 1 90); do
  kubectl logs "$POD" --timestamps --since=15s >> "$LOGFILE" 2>&1

  # Snapshot VRAM each cycle, as long as the container is still alive
  echo "$(date -u +%FT%TZ) --- rocm-smi snapshot ---" >> "$VRAMLOG"
  kubectl exec "$POD" -- rocm-smi >> "$VRAMLOG" 2>&1

  if grep -q "trainers.base_trainer\]\[INFO\] - step:" "$LOGFILE"; then
    RESULT="success"
    echo "$(date -u +%FT%TZ) *** SUCCESS on node $NODE (max_length=512 (XLA_FLAGS test)) ***" | tee -a "$LOGFILE"
    break
  fi

  TERM_STATE=$(kubectl get pod "$POD" -o jsonpath='{.status.containerStatuses[0].state.terminated}' 2>/dev/null)
  if [ -n "$TERM_STATE" ]; then
    RESULT="crashed"
    echo "$(date -u +%FT%TZ) *** CONTAINER TERMINATED on node $NODE: $TERM_STATE ***" | tee -a "$LOGFILE"
    kubectl logs "$POD" --timestamps >> "$LOGFILE" 2>&1
    kubectl describe pod "$POD" >> "$LOGFILE" 2>&1
    break
  fi

  sleep 5
done

if [ "$RESULT" = "unknown" ]; then
  echo "$(date -u +%FT%TZ) *** No step logged and no crash after ~15 minutes on node $NODE — treat as HUNG ***" | tee -a "$LOGFILE"
fi

echo "$(date -u +%FT%TZ) --- Final job status ---" | tee -a "$LOGFILE"
kubectl get job full-baseline -o wide >> "$LOGFILE" 2>&1
