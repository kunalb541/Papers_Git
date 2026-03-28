"""
Coherence Test — Pre-aggregated Adapter
Preregistration date: 2026-03-28

This adapter accepts results.json in the pre-aggregated format produced by
the evolution battery:

  pred_results:   list of {regime, target, level, score, ci_lo, ci_hi, n_reps}
  causal_results: list of {regime, target, interv_class, gap, ci_lo, ci_hi, ...}

DEVIATION FROM PREREGISTRATION (documented):
  The preregistration locked bootstrap CI estimation from raw replicate-level
  arrays. The actual results.json stores pre-aggregated scores with individual
  bootstrap CIs (not raw replicates). Pairwise gap CIs are therefore computed
  using the independence approximation:

    se_i  = (ci_hi_i - ci_lo_i) / (2 * z_alpha)
    se_ij = sqrt(se_i^2 + se_j^2)
    gap_ci = (gap_point ± z_alpha * se_ij)

  With n_reps = 100,000 per cell this approximation is tight. The deviation
  is recorded here and must be noted in any paper that reports these results.

All classification logic (coherence_label, winner_label, correspondence test,
pass/narrow_pass/fail criteria) is identical to coherence_test.py.
"""

import json
import numpy as np
from itertools import combinations, permutations
from scipy import stats
import argparse
import sys

# ── LOCKED CONSTANTS (identical to coherence_test.py) ────────────────────────

PREREG_DATE   = "2026-03-28"
BOOTSTRAP_ALPHA = 0.95
MIN_N         = 30
MIN_CLASSIFIED_FOR_INFORMATIVE = 6
WINNER_THRESHOLD  = 0.75
COUNTER_THRESHOLD = 0.25

DESCRIPTIONS = ["gene", "lineage", "organism", "organism_ecology"]
TASK_FAMILIES = ["predictive", "causal"]
REGIMES = [
    "selected_wf", "moran", "freq_dep",
    "eco_evol", "group_structured", "neutral_wf",
]
PREDICTIVE_TARGETS = ["persistence", "transmission", "dominance", "ecology"]
CAUSAL_TARGETS     = ["persistence", "transmission", "dominance", "ecology"]

Z_ALPHA = stats.norm.ppf(1 - (1 - BOOTSTRAP_ALPHA) / 2)   # 1.96 for 95%

# ── LOAD AND PIVOT ────────────────────────────────────────────────────────────

def load_and_pivot(path):
    """
    Load results.json and pivot flat lists into nested dicts.

    pred_scores[regime][desc][target]   = {"score", "ci_lo", "ci_hi", "n"}
    causal_scores[regime][desc][target] = {"score", "ci_lo", "ci_hi", "n"}

    'score' for causal = gap (intervention - control), matching preregistration
    definition of C(D;Y).
    """
    with open(path) as f:
        data = json.load(f)

    # Name translation: map results.json keys → locked DESCRIPTIONS/TARGETS
    DESC_MAP = {
        "org_eco": "organism_ecology",   # results.json uses org_eco
    }
    TGT_MAP = {
        # 'ecology' appearing as interv_class was a target label leak; no remap needed
        # Add any target renames here if discovered
    }

    pred_scores   = {r: {d: {} for d in DESCRIPTIONS} for r in REGIMES}
    causal_scores = {r: {d: {} for d in DESCRIPTIONS} for r in REGIMES}

    missing_pred_keys   = set()
    missing_causal_keys = set()

    for row in data.get("pred_results", []):
        regime = row.get("regime")
        desc   = DESC_MAP.get(row.get("level"), row.get("level"))
        target = row.get("target")
        if regime not in REGIMES:
            continue
        if desc not in DESCRIPTIONS:
            missing_pred_keys.add(desc)
            continue
        if target not in PREDICTIVE_TARGETS:
            continue
        n = row.get("n_reps", 0)
        if n < MIN_N or row.get("status") != "OK":
            continue
        pred_scores[regime][desc][target] = {
            "score":  row["score"],
            "ci_lo":  row["ci_lo"],
            "ci_hi":  row["ci_hi"],
            "n":      n,
        }

    for row in data.get("causal_results", []):
        regime = row.get("regime")
        desc   = DESC_MAP.get(row.get("interv_class"), row.get("interv_class"))
        target = row.get("target")
        if regime not in REGIMES:
            continue
        if desc not in DESCRIPTIONS:
            missing_causal_keys.add(desc)
            continue
        if target not in CAUSAL_TARGETS:
            continue
        n = min(row.get("n_ctrl", 0), row.get("n_int", 0))
        if n < MIN_N or row.get("status") != "OK":
            continue
        if row.get("contaminated", 0):
            continue
        causal_scores[regime][desc][target] = {
            "score":  row["gap"],          # gap = C(D;Y) per preregistration
            "ci_lo":  row["ci_lo"],
            "ci_hi":  row["ci_hi"],
            "n":      n,
        }

    if missing_pred_keys:
        print(f"WARNING: pred_results contains unrecognised level keys "
              f"(not in DESCRIPTIONS): {missing_pred_keys}")
    if missing_causal_keys:
        print(f"WARNING: causal_results contains unrecognised interv_class keys "
              f"(not in DESCRIPTIONS): {missing_causal_keys}")

    return pred_scores, causal_scores


def print_coverage(pred_scores, causal_scores):
    """Print how many cells are populated per regime/family."""
    print("\n  Coverage map (n estimable cells per regime):")
    print(f"  {'Regime':<22} {'Pred cells':<12} {'Causal cells'}")
    print(f"  {'-'*50}")
    for regime in REGIMES:
        p = sum(1 for d in DESCRIPTIONS for t in PREDICTIVE_TARGETS
                if pred_scores[regime][d].get(t))
        c = sum(1 for d in DESCRIPTIONS for t in CAUSAL_TARGETS
                if causal_scores[regime][d].get(t))
        print(f"  {regime:<22} {p:<12} {c}")
    print()

# ── PAIRWISE GAP TABLE ────────────────────────────────────────────────────────

def _gap_ci_from_precomputed(cell_i, cell_j):
    """
    Independence-approximation gap CI from pre-aggregated scores.
    Returns (lo, hi, point) or None if either cell missing.

    DEVIATION FROM PREREGISTRATION: uses independence approximation
    instead of joint replicate-level bootstrap.
    """
    if not cell_i or not cell_j:
        return None
    point = cell_i["score"] - cell_j["score"]
    se_i  = (cell_i["ci_hi"] - cell_i["ci_lo"]) / (2 * Z_ALPHA)
    se_j  = (cell_j["ci_hi"] - cell_j["ci_lo"]) / (2 * Z_ALPHA)
    se_ij = np.sqrt(se_i**2 + se_j**2)
    lo    = point - Z_ALPHA * se_ij
    hi    = point + Z_ALPHA * se_ij
    return float(lo), float(hi), float(point)


def compute_gap_table(scores, targets):
    """
    Precompute all pairwise gap CIs for one regime-family cell.
    scores[desc][target] = {"score", "ci_lo", "ci_hi", "n"}
    Returns gap_table[(di, dj, tgt)] = {"lo","hi","point","sign_label"}
    """
    gap_table = {}
    for (di, dj) in permutations(DESCRIPTIONS, 2):
        rev = (dj, di)
        for tgt in targets:
            key = (di, dj, tgt)
            rev_key = (dj, di, tgt)
            if rev_key in gap_table and gap_table[rev_key].get("lo") is not None:
                rev_entry = gap_table[rev_key]
                gap_table[key] = {
                    "lo":    -rev_entry["hi"],
                    "hi":    -rev_entry["lo"],
                    "point": -rev_entry["point"],
                    "sign_label": {"i_wins": "j_wins", "j_wins": "i_wins",
                                   "indeterminate": "indeterminate",
                                   "not_estimable": "not_estimable"}
                                  [rev_entry["sign_label"]],
                }
                continue
            ci_i = scores.get(di, {}).get(tgt)
            ci_j = scores.get(dj, {}).get(tgt)
            result = _gap_ci_from_precomputed(ci_i, ci_j)
            if result is None:
                gap_table[key] = {"sign_label": "not_estimable"}
                continue
            lo, hi, point = result
            if lo > 0:   sign = "i_wins"
            elif hi < 0: sign = "j_wins"
            else:        sign = "indeterminate"
            gap_table[key] = {"lo": lo, "hi": hi, "point": point,
                               "sign_label": sign}
    return gap_table

# ── COHERENCE CLASSIFICATION (identical logic to coherence_test.py) ───────────

def classify_coherence(gap_table, targets):
    pairs        = list(combinations(DESCRIPTIONS, 2))
    pair_labels  = {}
    pair_details = {}
    for (di, dj) in pairs:
        supported_signs = []
        detail = {}
        for tgt in targets:
            entry = gap_table.get((di, dj, tgt), {"sign_label": "not_estimable"})
            detail[tgt] = entry
            sl = entry["sign_label"]
            if sl not in ("indeterminate", "not_estimable"):
                supported_signs.append(sl)
        if len(supported_signs) == 0:
            pair_label = "undetermined"
        elif len(set(supported_signs)) == 1:
            pair_label = "sign_stable"
        else:
            pair_label = "sign_flipping"
        pair_labels[(di, dj)]  = pair_label
        pair_details[(di, dj)] = detail

    if any(v == "sign_flipping" for v in pair_labels.values()):
        cell_label = "incoherent"
    elif all(v == "sign_stable" for v in pair_labels.values()):
        cell_label = "coherent"
    else:
        cell_label = "unresolved"
    return cell_label, pair_labels, pair_details

# ── WINNER CLASSIFICATION (identical logic to coherence_test.py) ──────────────

def classify_winner(gap_table, scores, targets):
    estimable = [
        tgt for tgt in targets
        if all(scores.get(d, {}).get(tgt, {}).get("n", 0) >= MIN_N
               for d in DESCRIPTIONS)
    ]
    n_tgt = len(estimable)
    if n_tgt == 0:
        return None, "winner_absent", {"reason": "no_estimable_targets"}

    for d_star in DESCRIPTIONS:
        qualifies     = True
        winner_detail = {}
        for d_other in [d for d in DESCRIPTIONS if d != d_star]:
            sup_wins = sup_counters = 0
            for tgt in estimable:
                entry = gap_table.get((d_star, d_other, tgt),
                                      {"sign_label": "not_estimable"})
                if entry.get("lo") is not None:
                    if entry["lo"] > 0:  sup_wins     += 1
                    if entry["hi"] < 0:  sup_counters += 1
            win_frac = sup_wins     / n_tgt
            ctr_frac = sup_counters / n_tgt
            winner_detail[(d_star, d_other)] = {
                "supported_win_frac":     win_frac,
                "supported_counter_frac": ctr_frac,
            }
            if not (win_frac >= WINNER_THRESHOLD and
                    ctr_frac <= COUNTER_THRESHOLD):
                qualifies = False
                break
        if qualifies:
            return d_star, "stable_family_winner", winner_detail

    return None, "winner_absent", {}

# ── CORRESPONDENCE TEST ───────────────────────────────────────────────────────

def correspondence_test(coherence_labels, winner_labels):
    classified = []
    unresolved = []
    mismatches = []
    for regime in REGIMES:
        for family in TASK_FAMILIES:
            c = coherence_labels.get(regime, {}).get(family)
            w = winner_labels.get(regime, {}).get(family)
            if c is None or w is None:
                continue
            if c == "unresolved":
                unresolved.append((regime, family))
                continue
            expected_winner = (c == "coherent")
            actual_winner   = (w == "stable_family_winner")
            match = (expected_winner == actual_winner)
            classified.append({"regime": regime, "family": family,
                                "coherence": c, "winner": w, "match": match})
            if not match:
                mismatches.append((regime, family, c, w))

    n_classified = len(classified)
    if n_classified < MIN_CLASSIFIED_FOR_INFORMATIVE: verdict = "uninformative"
    elif len(mismatches) == 0:                        verdict = "pass"
    elif len(mismatches) == 1:                        verdict = "narrow_pass"
    else:                                             verdict = "fail"

    return {"verdict": verdict, "n_classified": n_classified,
            "n_unresolved": len(unresolved), "n_mismatch": len(mismatches),
            "classified_cells": classified, "unresolved_cells": unresolved,
            "mismatches": mismatches}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(data_path):
    print(f"\n{'='*60}")
    print(f"COHERENCE TEST (pre-aggregated adapter) -- {PREREG_DATE}")
    print(f"CI method: independence approximation (documented deviation)")
    print(f"{'='*60}")

    pred_scores, causal_scores = load_and_pivot(data_path)
    print_coverage(pred_scores, causal_scores)

    coherence_labels = {r: {} for r in REGIMES}
    winner_labels    = {r: {} for r in REGIMES}
    full_results     = {}

    for regime in REGIMES:
        full_results[regime] = {}
        for family in TASK_FAMILIES:
            scores  = pred_scores[regime]   if family == "predictive" else causal_scores[regime]
            targets = PREDICTIVE_TARGETS    if family == "predictive" else CAUSAL_TARGETS

            gap_table = compute_gap_table(scores, targets)

            coh_label, pair_labels, pair_details = classify_coherence(
                gap_table, targets)
            winner_desc, winner_label, winner_detail = classify_winner(
                gap_table, scores, targets)

            coherence_labels[regime][family] = coh_label
            winner_labels[regime][family]    = winner_label

            full_results[regime][family] = {
                "coherence_label": coh_label,
                "winner_label":    winner_label,
                "winner_desc":     winner_desc,
                "pair_labels":     {str(k): v for k, v in pair_labels.items()},
                "winner_detail":   {str(k): v for k, v in winner_detail.items()},
            }

            # Print sign-flip detail when incoherent
            flip_info = ""
            if coh_label == "incoherent":
                flips = [str(k) for k, v in pair_labels.items()
                         if v == "sign_flipping"]
                flip_info = f" [flips: {flips}]"
            print(f"  {regime:<22} {family:<12} "
                  f"coh={coh_label:<12} winner={winner_label}{flip_info}")

    corr = correspondence_test(coherence_labels, winner_labels)

    print(f"\n{'='*60}")
    print(f"CORRESPONDENCE TEST")
    print(f"{'='*60}")
    print(f"  Verdict         : {corr['verdict'].upper()}")
    print(f"  Classified cells: {corr['n_classified']}")
    print(f"  Unresolved cells: {corr['n_unresolved']}")
    print(f"  Mismatches      : {corr['n_mismatch']}")

    if corr['mismatches']:
        print(f"\n  Mismatches:")
        for m in corr['mismatches']:
            print(f"    {m[0]}/{m[1]}: coh={m[2]}, winner={m[3]}")

    print(f"\n  {'Regime':<22} {'Family':<12} {'Coherence':<14} "
          f"{'Winner':<26} Match")
    print(f"  {'-'*85}")
    for cell in corr['classified_cells']:
        print(f"  {cell['regime']:<22} {cell['family']:<12} "
              f"{cell['coherence']:<14} {cell['winner']:<26} "
              f"{'YES' if cell['match'] else 'NO'}")
    for r, f in corr['unresolved_cells']:
        print(f"  {r:<22} {f:<12} {'unresolved':<14} {'(excluded)':<26} -")

    out = {"prereg_date": PREREG_DATE, "ci_method": "independence_approximation",
           "verdict": corr["verdict"], "n_classified": corr["n_classified"],
           "n_mismatch": corr["n_mismatch"],
           "classified_cells": corr["classified_cells"],
           "unresolved_cells": [[r, f] for r, f in corr["unresolved_cells"]],
           "mismatches": [list(m) for m in corr["mismatches"]],
           "full_results": full_results}
    out_path = data_path.replace(".json", "_coherence_precomp.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to results.json (pre-aggregated format)")
    args = parser.parse_args()
    run(args.data)