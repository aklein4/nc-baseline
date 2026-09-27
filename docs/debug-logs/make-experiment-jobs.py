#!/usr/bin/env python3
"""Generate one job manifest per node from full-baseline.yaml.

We hold two granted nodes (bg-1, bg-2) and two competing hypotheses for the
step-11 stall, so run one on each and get both answers in a single window
instead of serialising two relaunches (each of which risks another node --
see docs/national-compute.md).

Derives from full-baseline.yaml rather than copying it, so the 300-line
startup script and its instrumentation stay in one place.

    python3 docs/debug-logs/make-experiment-jobs.py
    kubectl apply -f docs/debug-logs/job-mscl.yaml
    kubectl apply -f docs/debug-logs/job-cmdbuf.yaml
"""

import copy
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "full-baseline.yaml"
OUT = ROOT / "docs" / "debug-logs"

# (suffix, description, env overrides, XLA_FLAGS additions, extra env to strip)
# No node here on purpose: the marketplace chooses the node (see below).
EXPERIMENTS = [
    (
        "mscl",
        "Hypothesis 2: RCCL collective hang. RCCL_MSCCL_ENABLE=0 was "
        "independently sufficient to fix this exact signature in Part 7.",
        {"RCCL_MSCCL_ENABLE": "0"},
        None,
        set(),
    ),
    (
        "cmdbuf",
        "Hypothesis 4: the HIP-graph/command-buffer path. 'UpdateStreams "
        "failed' fired ~3,900/sec per graph launch right up to the hang and "
        "then stopped. XLA wraps collectives in command buffers, so disabling "
        "them tests whether that path is what wedges.",
        {},
        "--xla_gpu_enable_command_buffer=",
        set(),
    ),
    (
        "control",
        "THE CONTROL. Neither RCCL_MSCCL_ENABLE=0 nor command-buffers-off, "
        "everything else identical to the two runs that went clean (85 and "
        "148 steps). If this stalls, one or both of those changes is a real "
        "fix. If it runs clean past ~150 steps, neither is doing anything and "
        "the suspect becomes NCCL_DEBUG_SUBSYS=ALL perturbing the race timing "
        "-- i.e. the 'fix' is a Heisenbug mask. Note "
        "--xla_gpu_nccl_termination_timeout_seconds is NOT a candidate: it "
        "only terminates a stuck rendezvous, it cannot prevent one.",
        {},
        None,
        # Neither clean run had AITER_LOG_LEVEL (it was added afterwards), and
        # it changes stderr volume by ~570k lines -- which is exactly the
        # timing variable under suspicion. Strip it so this differs from those
        # runs by the intended variable only.
        {"AITER_LOG_LEVEL"},
    ),
]


def main() -> int:
    if not BASE.exists():
        print(f"missing {BASE}", file=sys.stderr)
        return 1
    base = yaml.safe_load(BASE.read_text())

    for suffix, why, env_extra, xla_extra, strip_extra in EXPERIMENTS:
        job = copy.deepcopy(base)
        name = f"full-baseline-{suffix}"
        job["metadata"]["name"] = name
        spec = job["spec"]["template"]["spec"]
        container = spec["containers"][0]

        # DO NOT pin with nodeSelector/nodeName. Kueue injects its own
        # selector on admission:
        #     marketplace.nationalcompute.com/request: <this grant's uuid>
        # and the marketplace relabels whichever node it assigns with that
        # uuid. A hostname pin therefore ANDs with the market's pin and, if
        # the market picks a different node than you guessed, the pod is
        # unschedulable forever ("0/6 nodes ... didn't match Pod's node
        # affinity/selector") -- which is exactly what happened at 02:32:
        # pinned to bg-2, market assigned bg-1.
        #
        # The market reuses nodes you already hold, so two unpinned jobs get
        # one node each. You cannot choose WHICH, and you do not need to.
        spec.pop("nodeSelector", None)

        env = container["env"]
        for key, value in env_extra.items():
            for entry in env:
                if entry.get("name") == key:
                    entry["value"] = value
                    break
            else:
                env.append({"name": key, "value": value})
        # Drop any env key this experiment must NOT carry, so the two runs
        # differ by exactly one variable.
        controlled = {"RCCL_MSCCL_ENABLE"} | strip_extra
        container["env"] = [
            e for e in env if e.get("name") not in (controlled - set(env_extra))
        ]

        if xla_extra:
            # The startup script builds EXTRA_XLA then exports XLA_FLAGS; append
            # ours to that same variable just before the export so the probe
            # results are preserved.
            args = container["args"][0]
            marker = 'echo "=== extra XLA flags for this run:'
            if marker not in args:
                print(f"marker not found in {BASE}; script changed?", file=sys.stderr)
                return 1
            args = args.replace(
                marker, f'EXTRA_XLA="$EXTRA_XLA {xla_extra}"\n              {marker}', 1
            )
            container["args"][0] = args

        header = (
            f"# GENERATED by docs/debug-logs/make-experiment-jobs.py -- do not edit.\n"
            f"# Edit full-baseline.yaml and regenerate.\n"
            f"#\n"
            f"# {name}\n"
            f"# {why}\n"
        )
        path = OUT / f"job-{suffix}.yaml"
        path.write_text(header + yaml.safe_dump(job, sort_keys=False, width=100))
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
