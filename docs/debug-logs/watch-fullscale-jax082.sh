#!/bin/bash
set -uo pipefail
LOGFILE="/Users/nivedithaiyer/Dev/adam/nc-baseline/docs/debug-logs/full-baseline-fullscale-jax082.log"
: > "$LOGFILE"

echo "$(date -u +%FT%TZ) Waiting for pod (FULL SCALE: max_length=1024, cluster_length=64, real data+init, JAX 0.8.2)..." | tee -a "$LOGFILE"

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

echo "$(date -u +%FT%TZ) --- Streaming logs with auto-reconnect (long run, no artificial timeout) ---" | tee -a "$LOGFILE"

while true; do
  STATE=$(kubectl get job full-baseline -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  if echo "$STATE" | grep -qE 'Complete|Failed'; then
    echo "$(date -u +%FT%TZ) Job reached terminal state: $STATE" | tee -a "$LOGFILE"
    break
  fi

  kubectl logs "$POD" --timestamps --since=30s >> "$LOGFILE" 2>&1

  PHASE=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
    echo "$(date -u +%FT%TZ) *** pod reached terminal phase: $PHASE ***" | tee -a "$LOGFILE"
    kubectl logs "$POD" --timestamps >> "$LOGFILE" 2>&1
    kubectl describe pod "$POD" >> "$LOGFILE" 2>&1
    break
  fi

  sleep 10
done

echo "$(date -u +%FT%TZ) --- Final job status ---" | tee -a "$LOGFILE"
kubectl get job full-baseline -o wide >> "$LOGFILE" 2>&1
