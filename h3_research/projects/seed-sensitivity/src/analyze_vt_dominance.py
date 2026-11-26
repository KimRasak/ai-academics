#!/usr/bin/env python3
"""Offline evidence for the proposed Sci-VBench VT reproducibility audit.

Uses only already-downloaded artifacts:
  - data/scivbench/vt_scores.jsonl   released per-video VT rows (11,216)
  - data/scivbench/testmini.jsonl    150 testmini records
  - data/scivbench/pilot16.jsonl     our frozen 16-prompt pilot split
  - data/scivbench/vt_full/h3_base/  our 32 fresh local VT records

No GPU, no network, no judge. Writes audits/round39_vt_dominance.json.
"""

import collections
import glob
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "scivbench")
METRICS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
CONTINUOUS = [m for m in METRICS if m != "dynamic_degree"]


def jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def kendall_tau(order_a, order_b):
    idx = {k: i for i, k in enumerate(order_b)}
    ks = [idx[k] for k in order_a]
    n = len(ks)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ks[i] < ks[j]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def bootstrap_ci(values, iters=20000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(iters))
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def main():
    released = list(jsonl(os.path.join(D, "vt_scores.jsonl")))
    testmini_ids = {r["id"] for r in jsonl(os.path.join(D, "testmini.jsonl"))}
    pilot_ids = [r["id"] for r in jsonl(os.path.join(D, "pilot16.jsonl"))]

    out = {"inputs": {"released_rows": len(released), "testmini": len(testmini_ids),
                      "pilot_prompts": len(pilot_ids)}}

    # --- A. composite definition check: VT is the unweighted mean of six metrics
    resid = [abs(st.mean(r[m] for m in METRICS) - r["VT"]) for r in released]
    out["composite_is_unweighted_mean"] = {"max_abs_residual": max(resid)}

    # --- B. between-model spread per metric (testmini, all 16 models)
    by_model = collections.defaultdict(list)
    for r in released:
        if r["video_id"] in testmini_ids:
            by_model[r["model"]].append(r)
    model_means = {
        k: {m: st.mean(x[m] for x in v) for m in METRICS} for k, v in by_model.items()
    }
    out["between_model_sd"] = {
        m: st.pstdev([model_means[k][m] for k in model_means]) for m in METRICS
    }
    # share of between-model VT variance carried by each term (VT = mean/6)
    tot = sum(out["between_model_sd"][m] ** 2 for m in METRICS)
    out["between_model_variance_share"] = {
        m: out["between_model_sd"][m] ** 2 / tot for m in METRICS
    }

    # --- C. leaderboard sensitivity to dropping dynamic_degree
    vt_rank = sorted(model_means, key=lambda k: -st.mean(by_model[k][i]["VT"]
                                                         for i in range(len(by_model[k]))))
    vt_rank = sorted(model_means,
                     key=lambda k: -st.mean(x["VT"] for x in by_model[k]))
    no_dd_rank = sorted(
        model_means,
        key=lambda k: -st.mean(st.mean(x[m] for m in CONTINUOUS) for x in by_model[k]),
    )
    out["leaderboard"] = {
        "vt_rank": vt_rank,
        "no_dd_rank": no_dd_rank,
        "kendall_tau": kendall_tau(vt_rank, no_dd_rank),
        "h3_rank_with_dd": vt_rank.index("MiniMax-H3") + 1,
        "h3_rank_without_dd": no_dd_rank.index("MiniMax-H3") + 1,
        "n_models": len(vt_rank),
    }

    # --- D. our fresh 32 local videos vs released MiniMax-H3 on the same 16 prompts
    rel_h3 = {r["video_id"]: r for r in released if r["model"] == "MiniMax-H3"}
    fresh = {}
    for path in sorted(glob.glob(os.path.join(D, "vt_full", "h3_base", "*.json"))):
        name = os.path.basename(path)[:-5]
        if "_seed" not in name:
            continue
        vid, seed = name.rsplit("_seed", 1)
        with open(path) as fh:
            fresh.setdefault(vid, {})[int(seed)] = json.load(fh)

    paired = {"n": 0, "vt_delta": [], "per_metric_delta": {m: [] for m in METRICS},
              "seed_abs_vt_gap": [], "dd_seed_flips": 0}
    for vid in pilot_ids:
        if vid not in rel_h3 or len(fresh.get(vid, {})) != 2:
            continue
        a, b, rel = fresh[vid][0], fresh[vid][1], rel_h3[vid]
        paired["n"] += 1
        paired["vt_delta"].append((a["VT"] + b["VT"]) / 2 - rel["VT"])
        for m in METRICS:
            paired["per_metric_delta"][m].append((a[m] + b[m]) / 2 - rel[m])
        paired["seed_abs_vt_gap"].append(abs(a["VT"] - b["VT"]))
        if a["dynamic_degree"] != b["dynamic_degree"]:
            paired["dd_seed_flips"] += 1

    lo, hi = bootstrap_ci(paired["vt_delta"])
    out["paired_reproduction"] = {
        "n_prompts": paired["n"],
        "vt_delta_mean": st.mean(paired["vt_delta"]),
        "vt_delta_sd": st.stdev(paired["vt_delta"]),
        "vt_delta_bootstrap95": [lo, hi],
        "per_metric_delta_mean": {m: st.mean(v) for m, v in paired["per_metric_delta"].items()},
        "per_metric_delta_sd": {m: st.stdev(v) for m, v in paired["per_metric_delta"].items()},
        "seed_abs_vt_gap_mean": st.mean(paired["seed_abs_vt_gap"]),
        "seed_abs_vt_gap_max": max(paired["seed_abs_vt_gap"]),
        "dd_seed_flip_rate": paired["dd_seed_flips"] / paired["n"],
    }

    # --- E. size of the gaps a benchmark claim would have to resolve
    testmini_vt = {k: st.mean(x["VT"] for x in v) for k, v in by_model.items()}
    ordered = sorted(testmini_vt.values(), reverse=True)
    gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]
    out["adjacent_vt_gaps"] = {
        "median": st.median(gaps), "min": min(gaps), "max": max(gaps),
        "h3_testmini_vt": testmini_vt["MiniMax-H3"],
    }

    dest = os.path.join(ROOT, "audits", "round39_vt_dominance.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {dest}")


if __name__ == "__main__":
    main()
