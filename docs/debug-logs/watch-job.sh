#!/bin/bash
# Stream one already-submitted job's logs to a file, with reconnect.
#
#   ./docs/debug-logs/watch-job.sh full-baseline-mscl
#   ./docs/debug-logs/watch-job.sh full-baseline-cmdbuf
#
# Attach-only by design: it never applies and never deletes. A delete would
# hand the node back to the marketplace and the next apply gets a fresh grant
# under a new request id (see docs/national-compute.md).
#
# Capture matters more than usual here: AMD_LOG_LEVEL=1 emits ~3,900 lines/sec
# of "UpdateStreams failed", so the kubelet's own log buffer rotates within
# seconds. This local file is the only durable copy.
set -uo pipefail

JOB="${1:?usage: watch-job.sh <job-name>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGFILE="$REPO/docs/debug-logs/${JOB}.log"
: > "$LOGFILE"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGFILE"; }

if ! kubectl get job "$JOB" >/dev/null 2>&1; then
  log "job $JOB does not exist -- nothing to attach to"
  exit 1
fi

POD=""
for _ in $(seq 1 180); do
  POD=$(kubectl get pods -l job-name="$JOB" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -n "$POD" ] && break
  sleep 10
done
[ -z "$POD" ] && { log "no pod for $JOB after ~30min"; exit 1; }

NODE=""
for _ in $(seq 1 30); do
  NODE=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  [ -n "$NODE" ] && break
  sleep 2
done
log "job=$JOB pod=$POD node=${NODE:-unassigned}"

# kubectl logs -f returns BadRequest immediately while a container is still
# ContainerCreating -- it does not block -- so wait rather than retry-spam.
for _ in $(seq 1 360); do
  CSTATE=$(kubectl get pod "$POD" \
    -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null)
  echo "$CSTATE" | grep -q '"waiting"' || break
  sleep 10
done
log "container state: ${CSTATE:-unknown}"

log "--- streaming (Ctrl-C to detach; the job keeps running) ---"
while true; do
  STATE=$(kubectl get job "$JOB" \
    -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  case "$STATE" in *Complete*|*Failed*) log "job terminal: $STATE"; break;; esac

  kubectl logs -f "$POD" --timestamps >> "$LOGFILE" 2>&1

  PHASE=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  case "$PHASE" in
    Succeeded|Failed)
      log "*** pod terminal: $PHASE ***"
      kubectl describe pod "$POD" >> "$LOGFILE" 2>&1
      break;;
    "") log "pod disappeared (reclaim?)"; break;;
  esac
  sleep 5
done

log "--- final status ---"
kubectl get job "$JOB" -o wide >> "$LOGFILE" 2>&1
