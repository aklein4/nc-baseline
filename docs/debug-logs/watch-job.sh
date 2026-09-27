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

# Capture the as-run spec immediately. The manifests get edited between
# launches, and the cluster applies them from the working tree rather than
# from a commit, so without this there is no way to tell afterwards what
# config a given run actually used -- and once the job is deleted the
# cluster cannot tell you either. Pull it from the API, not from the local
# file, so it is ground truth including anything the admission webhooks
# injected (queue name, suspend, the marketplace request selector).
ASRUN="$REPO/docs/debug-logs/${JOB}-AS-RUN.yaml"
kubectl get job "$JOB" -o yaml 2>/dev/null | python3 -c "
import sys, yaml
d = yaml.safe_load(sys.stdin)
if d:
    d['metadata'] = {k: v for k, v in d['metadata'].items() if k in ('name','labels')}
    d.pop('status', None)
    sys.stdout.write('# AS-RUN capture from the live cluster (kubectl get job -o yaml).\n')
    yaml.safe_dump(d, sys.stdout, sort_keys=False, width=100)
" > "$ASRUN" 2>/dev/null && log "as-run spec captured: ${ASRUN##*/}"

# Record the working-tree state too: the job clones src/ from GitHub at this
# commit, and a dirty tree means the YAML may not match anything committed.
{
  echo "# git HEAD at launch: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
  echo "# working tree: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ') modified path(s)"
} >> "$ASRUN" 2>/dev/null

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
