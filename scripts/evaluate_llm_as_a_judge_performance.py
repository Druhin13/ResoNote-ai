#!/usr/bin/env python3
"""
evaluate_llm_as_a_judge_performance.py

"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import math
import statistics as stats
import matplotlib.pyplot as plt

FACETS = ["Emotional_Tone", "Thematic_Content", "Narrative_Structure", "Lyrical_Style"]

# ------------------------- small plotting helpers -------------------------

def compute_figsize(rows, cols, cell_w=0.7, cell_h=0.55, min_w=8, min_h=6, max_w=42, max_h=60):
    w = min(max(min_w, cols * cell_w), max_w)
    h = min(max(min_h, rows * cell_h), max_h)
    return (w, h)

def save_bar(xlabels, values, title, ylabel, outfile, rotation=20):
    plt.figure(figsize=compute_figsize(10, len(xlabels), 0.6, 0.5))
    xs = list(range(len(xlabels)))
    plt.bar(xs, values)
    plt.xticks(xs, xlabels, rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def save_barh(labels, values, title, outfile):
    plt.figure(figsize=compute_figsize(len(labels), 8, 0.7, 0.55))
    y = list(range(len(labels)))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def save_hist(values, bins, title, xlabel, outfile):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def save_boxplot(groups, series_list, title, ylabel, outfile, rotation=0):
    plt.figure(figsize=compute_figsize(len(groups), 8, 0.6, 0.55))
    plt.boxplot(series_list, labels=groups, vert=True, showfliers=False)
    plt.xticks(rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def save_scatter(x, y, title, xlabel, ylabel, outfile):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

# ------------------------- loading & extraction -------------------------

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_decisions(eval_outputs):
    """
    Yields (track_id, facet, tag, agree, confidence, reason, suggested)
    """
    for rec in eval_outputs or []:
        tid = rec.get("track_id")
        parsed = (rec.get("parsed") or {})
        decisions = (parsed.get("decisions") or {})
        for facet in FACETS:
            for item in decisions.get(facet, []) or []:
                tag = item.get("tag")
                agree = bool(item.get("agree", False))
                conf = float(item.get("confidence", 0.0)) if isinstance(item.get("confidence"), (int, float, str)) else 0.0
                try:
                    conf = max(0.0, min(1.0, float(conf)))
                except Exception:
                    conf = 0.0
                reason = item.get("reason", "") or ""
                suggested = item.get("suggest_if_disagree", None)
                yield tid, facet, tag, agree, conf, reason, suggested

def per_track_agree_rate(eval_outputs):
    rates = []
    for rec in eval_outputs or []:
        parsed = (rec.get("parsed") or {})
        decisions = (parsed.get("decisions") or {})
        total = 0
        agrees = 0
        for f in FACETS:
            for it in decisions.get(f, []) or []:
                total += 1
                if it.get("agree") is True:
                    agrees += 1
        if total > 0:
            rates.append(agrees / total)
    return rates

# ------------------------- analysis -------------------------

def build_stats(eval_outputs):
    # Basic tallies
    per_facet_total = Counter({f: 0 for f in FACETS})
    per_facet_agree = Counter({f: 0 for f in FACETS})
    per_facet_autofill = Counter({f: 0 for f in FACETS})
    per_facet_suggest_ct = Counter({f: 0 for f in FACETS})

    conf_agree = []
    conf_disagree = []
    conf_agree_by_facet = defaultdict(list)
    conf_disagree_by_facet = defaultdict(list)
    reason_lengths = []

    # Per-tag acceptance across dataset
    tag_stats = {f: defaultdict(lambda: {"agree": 0, "total": 0}) for f in FACETS}
    suggested_counter = Counter()  # facet:tag → count

    for tid, facet, tag, agree, conf, reason, suggested in iter_decisions(eval_outputs):
        per_facet_total[facet] += 1
        if agree:
            per_facet_agree[facet] += 1
            conf_agree.append(conf)
            conf_agree_by_facet[facet].append(conf)
        else:
            conf_disagree.append(conf)
            conf_disagree_by_facet[facet].append(conf)

        if (reason or "").strip() == "not returned":
            per_facet_autofill[facet] += 1

        if suggested:
            per_facet_suggest_ct[facet] += 1
            suggested_counter[f"{facet}:{suggested}"] += 1

        # tag acceptance
        if tag:
            tag_stats[facet][tag]["total"] += 1
            if agree:
                tag_stats[facet][tag]["agree"] += 1

        reason_lengths.append(len(reason or ""))

    per_facet_agree_rate = {f: (per_facet_agree[f] / per_facet_total[f]) if per_facet_total[f] > 0 else 0.0 for f in FACETS}
    per_facet_autofill_rate = {f: (per_facet_autofill[f] / per_facet_total[f]) if per_facet_total[f] > 0 else 0.0 for f in FACETS}
    overall_agree_rate = (sum(per_facet_agree.values()) / max(1, sum(per_facet_total.values())))

    per_facet_tag_accept = {}
    for f in FACETS:
        items = []
        for tag, d in tag_stats[f].items():
            acc = d["agree"] / d["total"] if d["total"] > 0 else 0.0
            items.append((tag, acc, d["agree"], d["total"]))
        items.sort(key=lambda x: (-x[1], -x[3], x[0]))
        per_facet_tag_accept[f] = items

    return {
        "per_facet_total": per_facet_total,
        "per_facet_agree_rate": per_facet_agree_rate,
        "per_facet_autofill_rate": per_facet_autofill_rate,
        "overall_agree_rate": overall_agree_rate,
        "conf_agree": conf_agree,
        "conf_disagree": conf_disagree,
        "conf_agree_by_facet": conf_agree_by_facet,
        "conf_disagree_by_facet": conf_disagree_by_facet,
        "reason_lengths": reason_lengths,
        "suggested_counter": suggested_counter,
        "tag_accept": per_facet_tag_accept,
    }

def safe_pct(x):
    return f"{100.0 * x:.1f}%"

# ------------------------- figure set -------------------------

def make_figures(eval_outputs, eval_metrics, summaries_dir: Path, assets_dir: Path, top_k_tags=25):
    summaries_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    # Pull latency/tokens, trunc/cap flags if present
    latencies = [r.get("latency") for r in (eval_outputs or []) if isinstance(r.get("latency"), (int, float))]
    tok_in = [r.get("input_tokens_est") for r in (eval_outputs or []) if isinstance(r.get("input_tokens_est"), int)]
    tok_out = [r.get("output_tokens_est") for r in (eval_outputs or []) if isinstance(r.get("output_tokens_est"), int)]
    cap_hits = sum(1 for r in (eval_outputs or []) if r.get("hit_output_cap") is True)
    truncs = sum(1 for r in (eval_outputs or []) if r.get("truncated") is True)
    total_items = len(eval_outputs or [])

    # Core aggregates from decisions
    summary = build_stats(eval_outputs)
    rates_per_track = per_track_agree_rate(eval_outputs)

    manifest = []

    def put(fig_path: Path):
        manifest.append(str(fig_path))
        return str(fig_path)

    # 1) Per-track agree-rate histogram
    if rates_per_track:
        bins = [i/20 for i in range(21)]
        save_hist(rates_per_track, bins,
                  "Per-Track Agree Rate (judge vs candidate tags)", "agree rate",
                  put(assets_dir / "01_agree_rate_hist.png"))

    # 2) Per-facet agree rates
    save_bar(FACETS,
             [summary["per_facet_agree_rate"][f] for f in FACETS],
             "Per-Facet Agree Rate",
             "agree rate",
             put(assets_dir / "02_per_facet_agree_rate.png"),
             rotation=0)

    # 3) Confidence separation (agree vs disagree)
    if summary["conf_agree"] and summary["conf_disagree"]:
        save_boxplot(["agree", "disagree"],
                     [summary["conf_agree"], summary["conf_disagree"]],
                     "Confidence Separation (Agree vs Disagree)",
                     "confidence (0..1)",
                     put(assets_dir / "03_confidence_separation.png"))

    # 4) Confidence by facet & label
    for f in FACETS:
        agree_vals = summary["conf_agree_by_facet"][f]
        disagree_vals = summary["conf_disagree_by_facet"][f]
        if agree_vals or disagree_vals:
            groups, data = [], []
            if agree_vals:
                groups.append(f"{f}\nagree")
                data.append(agree_vals)
            if disagree_vals:
                groups.append(f"{f}\ndisagree")
                data.append(disagree_vals)
            if groups:
                save_boxplot(groups, data, f"Confidence by Label · {f}", "confidence (0..1)",
                             put(assets_dir / f"04_confidence_by_label_{f}.png"), rotation=0)

    # 5) Autofill rate by facet
    save_bar(FACETS,
             [summary["per_facet_autofill_rate"][f] for f in FACETS],
             "Autofill Rate (\"not returned\") by Facet",
             "fraction of decisions",
             put(assets_dir / "05_autofill_rate_by_facet.png"),
             rotation=0)

    # 6) Suggestion usage
    save_bar(FACETS,
             [sum(1 for k in summary["suggested_counter"].keys() if k.startswith(f + ":")) for f in FACETS],
             "Distinct Canonical Suggestions Emitted by Facet",
             "unique suggestions",
             put(assets_dir / "06_suggestion_unique_counts_by_facet.png"),
             rotation=0)

    # Top suggestions overall
    if summary["suggested_counter"]:
        top_sugg = summary["suggested_counter"].most_common(20)
        save_barh([k for k, _ in top_sugg],
                  [c for _, c in top_sugg],
                  "Most Common Suggestions (facet:tag)",
                  put(assets_dir / "07_top_suggestions.png"))

    # 7) Reason length adherence
    if summary["reason_lengths"]:
        bins = list(range(0, max(160, int(max(summary["reason_lengths"], default=0)) + 10), 10))
        save_hist(summary["reason_lengths"], bins, "Reason Length Distribution", "chars",
                  put(assets_dir / "08_reason_length_hist.png"))

    # 8) Latency distribution (if available)
    if latencies:
        save_hist(latencies, bins=20, title="Latency per Track", xlabel="seconds",
                  outfile=put(assets_dir / "09_latency_hist.png"))

    # 9) Token usage scatter (if available)
    if tok_in and tok_out and len(tok_in) == len(tok_out):
        save_scatter(tok_in, tok_out, "Token Use: Input vs Output per Track",
                     "input tokens (est)", "output tokens (est)",
                     put(assets_dir / "10_tokens_scatter.png"))

    # 10) Per-tag acceptance (top agreed & most rejected) per facet
    for f in FACETS:
        items = summary["tag_accept"][f]
        if not items:
            continue
        hi = [x for x in items if x[3] >= 3][:top_k_tags]
        if hi:
            save_barh([f"{t}  (n={n})" for t, acc, _, n in hi],
                      [acc for _, acc, __, ___ in hi],
                      f"Top Accepted Tags · {f}",
                      put(assets_dir / f"11_top_accepted_tags_{f}.png"))
        lo_full = sorted(items, key=lambda x: (x[1], -x[3], x[0]))
        lo = [x for x in lo_full if x[3] >= 3][:top_k_tags]
        if lo:
            save_barh([f"{t}  (n={n})" for t, acc, _, n in lo],
                      [acc for _, acc, __, ___ in lo],
                      f"Most Rejected Tags · {f}",
                      put(assets_dir / f"12_most_rejected_tags_{f}.png"))

    # Summary digest (in summaries_dir)
    digest = {
        "n_items": total_items,
        "truncate_hits": truncs,
        "output_cap_hits": cap_hits,
        "overall_agree_rate": round(summary["overall_agree_rate"], 4),
        "per_facet_agree_rate": {f: round(summary["per_facet_agree_rate"][f], 4) for f in FACETS},
        "per_facet_autofill_rate": {f: round(summary["per_facet_autofill_rate"][f], 4) for f in FACETS},
        "conf_agree_mean": round(stats.mean(summary["conf_agree"]), 4) if summary["conf_agree"] else None,
        "conf_disagree_mean": round(stats.mean(summary["conf_disagree"]), 4) if summary["conf_disagree"] else None,
        "latency_p50": (round(stats.median(latencies), 3) if latencies else None),
        "latency_p95": (round(sorted(latencies)[max(0, math.ceil(0.95*len(latencies))-1)], 3) if latencies else None),
        "tokens_avg_in": (round(sum(tok_in)/len(tok_in), 1) if tok_in else None),
        "tokens_avg_out": (round(sum(tok_out)/len(tok_out), 1) if tok_out else None),
    }
    if eval_metrics:
        for k in ["json_ok_rate","coverage_rate","cap_hit_fraction","avg_parsed_fraction","agree_rate","autofill_rate_overall"]:
            if k in eval_metrics:
                digest[k] = eval_metrics[k]

    with (summaries_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(digest, f, ensure_ascii=False, indent=2)

    # Figures manifest (paths relative to summaries_dir if possible)
    rel_manifest = []
    for p in manifest:
        try:
            rel = str(Path(p).relative_to(summaries_dir))
        except ValueError:
            rel = p
        rel_manifest.append(rel)
    with (summaries_dir / "figures_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"assets": rel_manifest}, f, ensure_ascii=False, indent=2)

    # Console summary
    print("==== LLM-as-a-judge confidence report ====")
    print(f"Tracks: {total_items} | Overall agree-rate: {round(summary['overall_agree_rate'], 4)}")
    print(f"Per-facet agree: " + ", ".join(f"{f}={safe_pct(summary['per_facet_agree_rate'][f])}" for f in FACETS))
    if digest.get("conf_agree_mean") is not None:
        print(f"Confidence (agree vs disagree): {digest['conf_agree_mean']} vs {digest['conf_disagree_mean']}")
    if digest.get("latency_p95") is not None:
        print(f"Latency p50/p95: {digest['latency_p50']}s / {digest['latency_p95']}s")
    if digest.get("tokens_avg_in") is not None:
        print(f"Tokens avg in/out: {digest['tokens_avg_in']} / {digest['tokens_avg_out']}")
    if "json_ok_rate" in digest:
        cap = digest.get("cap_hit_fraction", 0.0)
        print(f"Parser OK: {safe_pct(digest['json_ok_rate'])} | Coverage: {safe_pct(digest['coverage_rate'])} | Cap hits: {safe_pct(cap)}")
    print(f"Summary JSON : {summaries_dir / 'summary.json'}")
    print(f"Figures dir  : {assets_dir}")
    print(f"Manifest     : {summaries_dir / 'figures_manifest.json'}")

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_outputs", required=True, help="Path to eval_outputs.json from the judge run")
    ap.add_argument("--eval_metrics", default="", help="Optional: path to eval_metrics.json")
    ap.add_argument("--summaries_dir", default="analysis/llm_as_a_judge/summaries",
                    help="Directory to store summaries (JSON). Default: analysis/llm_as_a_judge/summaries")
    ap.add_argument("--assets_dir", default="",
                    help="Directory for figures (PNG). Default: <summaries_dir>/assets")
    ap.add_argument("--top_k_tags", type=int, default=25)
    args = ap.parse_args()

    eval_outputs = load_json(args.eval_outputs)
    if not isinstance(eval_outputs, list):
        raise ValueError("eval_outputs.json must be a list of records.")
    eval_metrics = load_json(args.eval_metrics) if args.eval_metrics else None

    summaries_dir = Path(args.summaries_dir)
    assets_dir = Path(args.assets_dir) if args.assets_dir else summaries_dir / "assets"

    make_figures(eval_outputs, eval_metrics, summaries_dir, assets_dir, top_k_tags=args.top_k_tags)

if __name__ == "__main__":
    main()
