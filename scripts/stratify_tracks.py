#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import defaultdict, Counter
from pathlib import Path

def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be an array")
    return data

def save_json_array(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def dedupe_by_key(rows, key):
    seen = set()
    out = []
    for r in rows:
        k = r.get(key)
        if k is None:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def valid_number(x):
    try:
        if x is None:
            return False
        float(x)
        return True
    except:
        return False

def bindex(v, edges):
    for i in range(len(edges) - 1):
        if i < len(edges) - 2:
            if edges[i] <= v < edges[i + 1]:
                return i
        else:
            if edges[i] <= v <= edges[i + 1]:
                return i
    return None

def apportion(counts, total_target, min_per_bucket=1):
    total_items = sum(counts.values())
    buckets = list(counts.keys())
    targets = {}
    floors = {}
    for b in buckets:
        t = (counts[b] / total_items) * total_target if total_items > 0 else 0
        floors[b] = min(counts[b], math.floor(t))
        targets[b] = t
    alloc = {b: max(min_per_bucket if counts[b] > 0 else 0, floors[b]) for b in buckets}
    cap = {b: counts[b] - alloc[b] for b in buckets}
    s = sum(alloc.values())
    if s > total_target:
        order = sorted(buckets, key=lambda b: (alloc[b] - floors[b], targets[b]))
        i = 0
        while s > total_target and i < len(order):
            b = order[i]
            if alloc[b] > 0 and (alloc[b] - floors[b]) > 0:
                alloc[b] -= 1
                s -= 1
            i += 1
        if s > total_target:
            order = sorted(buckets, key=lambda b: alloc[b], reverse=True)
            i = 0
            while s > total_target and i < len(order):
                b = order[i]
                if alloc[b] > 0:
                    alloc[b] -= 1
                    s -= 1
                i += 1
    elif s < total_target:
        leftovers = total_target - s
        fracs = {b: targets[b] - math.floor(targets[b]) for b in buckets}
        order = sorted(buckets, key=lambda b: (fracs[b], counts[b]), reverse=True)
        i = 0
        while leftovers > 0 and any(cap[b] > 0 for b in buckets):
            b = order[i % len(order)]
            if cap[b] > 0:
                alloc[b] += 1
                cap[b] -= 1
                leftovers -= 1
            i += 1
    return alloc

def summarize(selected, v_edges, p_edges):
    v_counts = Counter()
    p_counts = Counter()
    grid = defaultdict(int)
    for r in selected:
        v = float(r.get("valence", 0))
        p = float(r.get("popularity", 0))
        vi = bindex(v, v_edges)
        pi = bindex(p, p_edges)
        if vi is not None:
            v_counts[vi] += 1
        if pi is not None:
            p_counts[pi] += 1
        if vi is not None and pi is not None:
            grid[(vi, pi)] += 1
    return v_counts, p_counts, grid

def print_report(total_in, total_kept, v_edges, p_edges, v_counts, p_counts, grid):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
        console.print(Panel.fit(f"Stratified Sample Summary\nInput: {total_in} | Selected: {total_kept}", title="ResoNote Stratifier", border_style="green"))
        tv = Table(title="Valence Distribution", box=box.SIMPLE)
        tv.add_column("Bin")
        tv.add_column("Range")
        tv.add_column("Count", justify="right")
        for i in range(len(v_edges) - 1):
            lo = v_edges[i]
            hi = v_edges[i + 1]
            rng = f"[{lo:.2f}, {hi:.2f}{']' if i==len(v_edges)-2 else ')'}"
            tv.add_row(str(i), rng, str(v_counts.get(i, 0)))
        console.print(tv)
        tp = Table(title="Popularity Distribution", box=box.SIMPLE)
        tp.add_column("Bin")
        tp.add_column("Range")
        tp.add_column("Count", justify="right")
        for i in range(len(p_edges) - 1):
            lo = p_edges[i]
            hi = p_edges[i + 1]
            rng = f"[{int(lo)}, {int(hi)}{']' if i==len(p_edges)-2 else ')'}"
            tp.add_row(str(i), rng, str(p_counts.get(i, 0)))
        console.print(tp)
        tg = Table(title="Grid Counts (valence × popularity)", box=box.SIMPLE)
        tg.add_column("Valence Bin")
        tg.add_column("Popularity Bin")
        tg.add_column("Count", justify="right")
        for i in range(len(v_edges) - 1):
            for j in range(len(p_edges) - 1):
                if grid.get((i, j), 0) > 0:
                    tg.add_row(str(i), str(j), str(grid[(i, j)]))
        console.print(tg)
    except Exception:
        print(f"Input: {total_in} | Selected: {total_kept}")
        print("Valence Distribution:")
        for i in range(len(v_edges) - 1):
            lo = v_edges[i]
            hi = v_edges[i + 1]
            rng = f"[{lo:.2f}, {hi:.2f}{']' if i==len(v_edges)-2 else ')'}"
            print(f"- Bin {i} {rng}: {v_counts.get(i,0)}")
        print("Popularity Distribution:")
        for i in range(len(p_edges) - 1):
            lo = p_edges[i]
            hi = p_edges[i + 1]
            rng = f"[{int(lo)}, {int(hi)}{']' if i==len(p_edges)-2 else ')'}"
            print(f"- Bin {i} {rng}: {p_counts.get(i,0)}")
        print("Grid Counts:")
        for i in range(len(v_edges) - 1):
            for j in range(len(p_edges) - 1):
                c = grid.get((i, j), 0)
                if c > 0:
                    print(f"- v{i} × p{j}: {c}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/processed/track_data/audio_features_and_lyrics_cleaned.json")
    ap.add_argument("--output", default="dataset/stratified/250_sample_tracks.json")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--valence-bins", type=int, default=5)
    ap.add_argument("--popularity-bins", type=int, default=5)
    args = ap.parse_args()
    random.seed(args.seed)
    rows = load_json_array(Path(args.input))
    rows = dedupe_by_key(rows, "track_id")
    rows = [r for r in rows if valid_number(r.get("valence")) and valid_number(r.get("popularity"))]
    v_edges = [i / args.valence_bins for i in range(args.valence_bins + 1)]
    p_step = 100 / args.popularity_bins
    p_edges = [i * p_step for i in range(args.popularity_bins)] + [100]
    p_edges[-1] = 100
    p_edges = [float(x) for x in p_edges]
    buckets = defaultdict(list)
    for r in rows:
        v = float(r.get("valence"))
        p = float(r.get("popularity"))
        vi = bindex(v, v_edges + [v_edges[-1] + 1e-9])
        pi = bindex(p, p_edges + [p_edges[-1] + 1e-9])
        if vi is None or pi is None:
            continue
        buckets[(vi, pi)].append(r)
    nonempty = {k: v for k, v in buckets.items() if len(v) > 0}
    counts = {k: len(v) for k, v in nonempty.items()}
    if sum(counts.values()) < args.n:
        raise RuntimeError("Not enough records to sample the requested size")
    alloc = apportion(counts, args.n, min_per_bucket=1 if len(nonempty) <= args.n else 0)
    selected = []
    for k, al in alloc.items():
        c = buckets[k]
        if al <= 0:
            continue
        if al >= len(c):
            chosen = list(c)
        else:
            chosen = random.sample(c, al)
        selected.extend(chosen)
    if len(selected) < args.n:
        pool_ids = set(r.get("track_id") for r in selected)
        remaining = [r for r in rows if r.get("track_id") not in pool_ids]
        need = args.n - len(selected)
        if need > 0:
            selected.extend(random.sample(remaining, need))
    selected = selected[: args.n]
    save_json_array(Path(args.output), selected)
    v_counts, p_counts, grid = summarize(selected, v_edges, p_edges)
    print_report(len(rows), len(selected), v_edges, p_edges, v_counts, p_counts, grid)

if __name__ == "__main__":
    main()
