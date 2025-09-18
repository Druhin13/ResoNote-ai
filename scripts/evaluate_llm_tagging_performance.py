#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from statistics import mean
from pathlib import Path
import matplotlib.pyplot as plt

FACETS = ["Emotional_Tone", "Thematic_Content", "Narrative_Structure", "Lyrical_Style"]

def normalize_tag(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    out = []
    prev_dash = False
    for ch in s.replace("_", " "):
        if ch.isspace() or ch == "-":
            if not prev_dash:
                out.append("-")
            prev_dash = True
        else:
            out.append(ch)
            prev_dash = False
    t = "".join(out).strip("-")
    while "--" in t:
        t = t.replace("--", "-")
    return t

def load_canonical(path: str):
    if not path or not Path(path).exists():
        return {f: None for f in FACETS}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {f: set(normalize_tag(x) for x in obj.get(f, [])) for f in FACETS}

def load_gold(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gold = {}
    names = {}
    for item in data.get("annotations", []):
        tid = str(item.get("track_id"))
        sel = item.get("selections", {})
        gold[tid] = {facet: set(normalize_tag(t) for t in sel.get(facet, [])) for facet in FACETS}
        names[tid] = item.get("track_name", "")
    return gold, names

def load_model_predictions(fp: str):
    name = Path(fp).stem.replace("eval_tags_", "")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("annotations", [])
    preds = {}
    for rec in records:
        tid = str(rec.get("track_id"))
        block = rec.get("tags") or rec.get("selections") or {}
        preds[tid] = {facet: set(normalize_tag(t) for t in block.get(facet, [])) for facet in FACETS}
    usage = {}
    if isinstance(data, dict) and "usage" in data:
        usage = data["usage"] or {}
    return name, preds, usage

def prf1_jacc(tp: int, fp: int, fn: int, union: int):
    p = tp / (tp + fp) if tp + fp > 0 else 1.0 if tp + fp + fn == 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 1.0 if tp + fp + fn == 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    j = tp / union if union > 0 else 1.0
    return p, r, f1, j

def evaluate(gold, preds, canonical):
    tracks = sorted(gold.keys())
    per_facet_macro = {}
    per_facet_micro = {}
    per_facet_counts = {}
    per_facet_tp_fp_fn = {}
    fp_tags = {facet: Counter() for facet in FACETS}
    fn_tags = {facet: Counter() for facet in FACETS}
    coverage_tracks = set()
    for facet in FACETS:
        per_track_scores = []
        tp = fp = fn = union = 0
        for tid in tracks:
            G = set(gold[tid].get(facet, set()))
            P = set(preds.get(tid, {}).get(facet, set()))
            if preds.get(tid) is not None:
                coverage_tracks.add(tid)
            if canonical.get(facet) is not None:
                G = G & canonical[facet]
                P = P & canonical[facet]
            tpi = len(G & P)
            fpi = len(P - G)
            fni = len(G - P)
            uni = len(G | P)
            p, r, f1, j = prf1_jacc(tpi, fpi, fni, uni)
            per_track_scores.append((p, r, f1, j))
            tp += tpi
            fp += fpi
            fn += fni
            union += uni
            for t in (P - G):
                fp_tags[facet][t] += 1
            for t in (G - P):
                fn_tags[facet][t] += 1
        if per_track_scores:
            per_facet_macro[facet] = {
                "precision": round(mean(x[0] for x in per_track_scores), 4),
                "recall": round(mean(x[1] for x in per_track_scores), 4),
                "f1": round(mean(x[2] for x in per_track_scores), 4),
                "jaccard": round(mean(x[3] for x in per_track_scores), 4),
                "tracks": len(per_track_scores),
            }
        else:
            per_facet_macro[facet] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "tracks": 0}
        Pm, Rm, Fm, Jm = prf1_jacc(tp, fp, fn, union)
        per_facet_micro[facet] = {
            "precision": round(Pm, 4),
            "recall": round(Rm, 4),
            "f1": round(Fm, 4),
            "jaccard": round(Jm, 4),
        }
        per_facet_counts[facet] = {"tp": tp, "fp": fp, "fn": fn, "union": union}
        per_facet_tp_fp_fn[facet] = (tp, fp, fn)
    tp_all = sum(per_facet_tp_fp_fn[f][0] for f in FACETS)
    fp_all = sum(per_facet_tp_fp_fn[f][1] for f in FACETS)
    fn_all = sum(per_facet_tp_fp_fn[f][2] for f in FACETS)
    union_all = sum(per_facet_counts[f]["union"] for f in FACETS)
    P_all, R_all, F_all, J_all = prf1_jacc(tp_all, fp_all, fn_all, union_all)
    overall_macro_f1 = round(mean(per_facet_macro[f]["f1"] for f in FACETS), 4)
    overall = {
        "micro_precision": round(P_all, 4),
        "micro_recall": round(R_all, 4),
        "overall_micro_f1": round(F_all, 4),
        "overall_jaccard": round(J_all, 4),
        "overall_macro_f1": overall_macro_f1,
    }
    return per_facet_macro, per_facet_micro, per_facet_counts, overall, fp_tags, fn_tags, coverage_tracks

def inter_model_agreement(preds_by_model, subset_track_ids):
    out = {facet: {} for facet in FACETS}
    model_names = sorted(preds_by_model.keys())
    for facet in FACETS:
        sets_by_model = {}
        for m in model_names:
            s = set()
            for tid, sel in preds_by_model[m].items():
                if tid in subset_track_ids:
                    for t in sel.get(facet, []):
                        s.add((tid, t))
            sets_by_model[m] = s
        for a in model_names:
            out[facet][a] = {}
            for b in model_names:
                A = sets_by_model[a]
                B = sets_by_model[b]
                inter = len(A & B)
                uni = len(A | B)
                jacc = inter / uni if uni > 0 else 1.0
                out[facet][a][b] = jacc
    return out, model_names

def compute_figsize(rows, cols, cell_w=0.6, cell_h=0.5, min_w=8, min_h=6, max_w=40, max_h=60):
    w = min(max(min_w, cols * cell_w), max_w)
    h = min(max(min_h, rows * cell_h), max_h)
    return (w, h)

def bar_chart(xlabels, values, title, ylabel, outfile, rotation=30):
    plt.figure(figsize=compute_figsize(10, len(xlabels), 0.6, 0.5))
    xs = list(range(len(xlabels)))
    plt.bar(xs, values)
    plt.xticks(xs, xlabels, rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def grouped_bar(models, groups, matrix, title, ylabel, outfile, group_sep=0.7):
    n_models = len(models)
    n_groups = len(groups)
    width = 0.8 / max(1, n_models)
    total_cols = n_groups * n_models
    plt.figure(figsize=compute_figsize(10, total_cols, 0.35, 0.5))
    positions = []
    ticks = []
    cur = 0.0
    for g in range(n_groups):
        base = cur
        for m in range(n_models):
            x = base + m * width
            positions.append(x)
        ticks.append(base + (n_models - 1) * width / 2)
        cur = base + n_models * width + group_sep
    heights = []
    for g in range(n_groups):
        for m in range(n_models):
            heights.append(matrix[m][g])
    plt.bar(positions, heights)
    plt.xticks(ticks, groups, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def heatmap(xlabels, ylabels, matrix, title, outfile, annotate_threshold=300):
    rows = len(ylabels)
    cols = len(xlabels)
    plt.figure(figsize=compute_figsize(rows, cols, 0.7, 0.55, 10, 6, 48, 60))
    ax = plt.gca()
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(cols))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(rows))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    cb = plt.colorbar(im)
    cb.ax.set_ylabel("value")
    if rows * cols <= annotate_threshold:
        for i in range(rows):
            for j in range(cols):
                val = matrix[i][j]
                txt = f"{val:.2f}" if isinstance(val, float) else str(val)
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(outfile, dpi=240, bbox_inches="tight")
    plt.close()

def build_per_tag_sets(gold, preds_by_model, canonical):
    entities = ["Human"] + sorted(preds_by_model.keys())
    per_facet = {}
    for facet in FACETS:
        per_facet[facet] = {}
        for ent in entities + ["AnyModel"]:
            per_facet[facet][ent] = defaultdict(set)
        for tid, sels in gold.items():
            tags = sels.get(facet, set())
            if canonical.get(facet) is not None:
                tags = tags & canonical[facet]
            for t in tags:
                per_facet[facet]["Human"][t].add(tid)
        for m, preds in preds_by_model.items():
            for tid, sels in preds.items():
                if tid not in gold:
                    continue
                tags = sels.get(facet, set())
                if canonical.get(facet) is not None:
                    tags = tags & canonical[facet]
                for t in tags:
                    per_facet[facet][m][t].add(tid)
        all_tags = set(per_facet[facet]["Human"].keys())
        for m in preds_by_model.keys():
            all_tags |= set(per_facet[facet][m].keys())
        for tag in all_tags:
            u = set()
            for m in preds_by_model.keys():
                u |= per_facet[facet][m].get(tag, set())
            per_facet[facet]["AnyModel"][tag] = u
    return per_facet

def per_tag_union_media(per_tag_sets, assets_dir, top_k=30):
    for facet in FACETS:
        human_counts = {tag: len(per_tag_sets[facet]["Human"].get(tag, set())) for tag in per_tag_sets[facet]["Human"].keys()}
        tags_sorted = sorted(human_counts.items(), key=lambda x: x[1], reverse=True)
        tags_top = [t for t, c in tags_sorted[:top_k]] if tags_sorted else []
        entities = ["Human"] + [e for e in per_tag_sets[facet].keys() if e not in ("Human", "AnyModel")] + ["AnyModel"]
        matrix_counts = []
        for tag in tags_top:
            row = []
            for ent in entities:
                row.append(len(per_tag_sets[facet][ent].get(tag, set())))
            matrix_counts.append(row)
        heatmap(entities, tags_top, matrix_counts, f"Per-Tag Coverage Counts · {facet}", str(assets_dir / f"per_tag_counts_heatmap_{facet}.png"))
        models_only = [e for e in entities if e not in ("Human", "AnyModel")]
        matrix_jacc = []
        for tag in tags_top:
            hset = per_tag_sets[facet]["Human"].get(tag, set())
            row = []
            for ent in models_only:
                eset = per_tag_sets[facet][ent].get(tag, set())
                inter = len(hset & eset)
                uni = len(hset | eset)
                j = inter / uni if uni > 0 else 1.0
                row.append(j)
            matrix_jacc.append(row)
        heatmap(models_only, tags_top, matrix_jacc, f"Per-Tag Jaccard(Human vs Model) · {facet}", str(assets_dir / f"per_tag_jaccard_heatmap_{facet}.png"))

def barh_chart(labels, values, title, outfile):
    plt.figure(figsize=compute_figsize(len(labels), 10, 0.7, 0.55, 10, 6, 40, 60))
    y = list(range(len(labels)))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close()

def best_model_error_bars(best_model, per_model_fn, per_model_fp, top_k, assets_dir):
    for facet in FACETS:
        fns = per_model_fn[best_model][facet].most_common(top_k)
        fps = per_model_fp[best_model][facet].most_common(top_k)
        if fns:
            barh_chart([t for t, c in fns], [c for t, c in fns], f"{best_model} · Top Missed Human Tags (FN) · {facet}", str(assets_dir / f"{best_model}_top_fn_{facet}.png"))
        if fps:
            barh_chart([t for t, c in fps], [c for t, c in fps], f"{best_model} · Top Model-Only Tags (FP) · {facet}", str(assets_dir / f"{best_model}_top_fp_{facet}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="analysis/human_evaluation/resonote_app_annotations.json")
    ap.add_argument("--models_glob", default="analysis/llm_selection_p1/final_tags/eval_tags_*.json")
    ap.add_argument("--canonical", default="")
    ap.add_argument("--out_json", default="analysis/human_evaluation/summaries/llm_selection_results.json")
    ap.add_argument("--assets_dir", default="analysis/human_evaluation/summaries/assets")
    ap.add_argument("--top_k_errors", type=int, default=15)
    ap.add_argument("--per_tag_top_k", type=int, default=30)
    args = ap.parse_args()
    Path(args.assets_dir).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 15, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    canonical = load_canonical(args.canonical) if args.canonical else {f: None for f in FACETS}
    gold, track_names = load_gold(args.gold)
    gold_track_ids = set(gold.keys())
    model_files = sorted(glob.glob(args.models_glob))
    results = {"models": {}, "ranking": [], "best_model": "", "assets_dir": str(Path(args.assets_dir))}
    preds_by_model = {}
    usage_by_model = {}
    for mf in model_files:
        model_name, preds, usage = load_model_predictions(mf)
        usage_by_model[model_name] = usage
        preds_by_model[model_name] = preds
    per_model_overall = []
    per_model_overall_precision = []
    per_model_overall_recall = []
    per_model_track_coverage = []
    per_model_perfacet_micro = defaultdict(dict)
    per_model_perfacet_macro = defaultdict(dict)
    per_model_counts = defaultdict(dict)
    per_model_fp = defaultdict(lambda: defaultdict(Counter))
    per_model_fn = defaultdict(lambda: defaultdict(Counter))
    for model_name in sorted(preds_by_model.keys()):
        macro, micro, counts, overall, fp_tags, fn_tags, coverage_tracks = evaluate(gold, preds_by_model[model_name], canonical)
        results["models"][model_name] = {
            "tracks_evaluated": len(gold),
            "per_facet": {f: {"macro": macro[f], "micro": micro[f]} for f in FACETS},
            "overall": overall,
            "source_file": [mf for mf in model_files if model_name in mf][0] if model_files else ""
        }
        per_model_overall.append((model_name, overall["overall_micro_f1"]))
        per_model_overall_precision.append((model_name, overall["micro_precision"]))
        per_model_overall_recall.append((model_name, overall["micro_recall"]))
        per_model_track_coverage.append((model_name, len(coverage_tracks)))
        for f in FACETS:
            per_model_perfacet_micro[model_name][f] = micro[f]["f1"]
            per_model_perfacet_macro[model_name][f] = macro[f]["f1"]
            per_model_counts[model_name][f] = counts[f]
            per_model_fp[model_name][f] = fp_tags[f]
            per_model_fn[model_name][f] = fn_tags[f]
    ranking = sorted(
        [{"model": m, "overall_micro_f1": sc} for m, sc in per_model_overall],
        key=lambda x: x["overall_micro_f1"],
        reverse=True,
    )
    results["ranking"] = ranking
    results["best_model"] = ranking[0]["model"] if ranking else ""
    Path(os.path.dirname(args.out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    models_sorted = [m for m, _ in sorted(per_model_overall, key=lambda x: x[1], reverse=True)]
    bar_chart(models_sorted, [dict(per_model_overall)[m] for m in models_sorted], "Overall Micro-F1 by Model", "micro-F1", str(Path(args.assets_dir) / "overall_micro_f1_by_model.png"))
    bar_chart(models_sorted, [dict(per_model_overall_precision)[m] for m in models_sorted], "Overall Micro-Precision by Model", "precision", str(Path(args.assets_dir) / "overall_micro_precision_by_model.png"))
    bar_chart(models_sorted, [dict(per_model_overall_recall)[m] for m in models_sorted], "Overall Micro-Recall by Model", "recall", str(Path(args.assets_dir) / "overall_micro_recall_by_model.png"))
    bar_chart(models_sorted, [dict(per_model_track_coverage)[m] for m in models_sorted], "Track Coverage on 50 Gold Tracks", "tracks with predictions", str(Path(args.assets_dir) / "track_coverage.png"))
    micro_matrix = []
    macro_matrix = []
    for m in models_sorted:
        micro_matrix.append([per_model_perfacet_micro[m][f] for f in FACETS])
        macro_matrix.append([per_model_perfacet_macro[m][f] for f in FACETS])
    grouped_bar(models_sorted, FACETS, micro_matrix, "Per-Facet Micro-F1", "micro-F1", str(Path(args.assets_dir) / "per_facet_micro_f1_grouped.png"))
    grouped_bar(models_sorted, FACETS, macro_matrix, "Per-Facet Macro-F1", "macro-F1", str(Path(args.assets_dir) / "per_facet_macro_f1_grouped.png"))
    agree, model_names = inter_model_agreement(preds_by_model, gold_track_ids)
    for facet in FACETS:
        mat = []
        for a in model_names:
            row = []
            for b in model_names:
                row.append(agree[facet][a][b])
            mat.append(row)
        heatmap(model_names, model_names, mat, f"Inter-Model Jaccard Agreement · {facet}", str(Path(args.assets_dir) / f"inter_model_agreement_{facet}.png"))
    per_tag_sets = build_per_tag_sets(gold, preds_by_model, canonical)
    per_tag_union_media(per_tag_sets, Path(args.assets_dir), top_k=args.per_tag_top_k)
    if results["best_model"]:
        best_model_error_bars(results["best_model"], per_model_fn, per_model_fp, args.top_k_errors, Path(args.assets_dir))
    print(args.out_json)
    print(str(Path(args.assets_dir)))

if __name__ == "__main__":
    main()
