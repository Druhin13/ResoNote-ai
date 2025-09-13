#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

def to_int_ms(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except:
        return None

def tokenize(s):
    if not isinstance(s, str):
        return []
    return re.findall(r"[a-z0-9]+", s.lower())

def has_mixlike_title_flags(title_tokens):
    phrases = [
        ("dj", "mix"),
        ("continuous", "mix"),
        ("nonstop", "mix"),
        ("non", "stop", "mix"),
        ("mega", "mix"),
        ("megamix",),
        ("mix",), 
        ("mixtape",),
        ("playlist",),
        ("dj", "set"),
        ("live", "set"),
        ("album", "mix"),
        ("full", "mix"),
    ]
    tset = set(title_tokens)
    title_join = " ".join(title_tokens)
    flags = []
    if "instrumental" in tset:
        flags.append("instrumental_in_title")
    for p in phrases:
        if len(p) == 1:
            if p[0] in tset:
                if p[0] == "mix":
                    if "remix" in tset or "remixed" in tset:
                        continue
                flags.append("mixlike_title")
        else:
            if " ".join(p) in title_join:
                flags.append("mixlike_title")
    return list(set(flags))

def has_instrumental_in_lyrics(lyrics):
    if not isinstance(lyrics, str):
        return False
    return re.search(r"\binstrumental\b", lyrics.lower()) is not None

def normalize_lyrics_for_length(lyrics):
    if not isinstance(lyrics, str):
        return ""
    s = re.sub(r"\[[^\]]+\]", " ", lyrics)
    s = re.sub(r"\([^\)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def lyrics_too_short(lyrics, min_words, min_chars):
    if not isinstance(lyrics, str) or lyrics.strip() == "":
        return True
    norm = normalize_lyrics_for_length(lyrics)
    words = re.findall(r"[A-Za-z']+", norm)
    char_count = sum(len(w) for w in words)
    return len(words) < min_words or char_count < min_chars

def evaluate_reasons(track, cfg):
    reasons = []
    dur = to_int_ms(track.get("duration_ms"))
    if dur is not None:
        if dur < cfg["min_ms"]:
            reasons.append("short_duration")
        if dur > cfg["max_ms"]:
            reasons.append("long_duration")
    title_tokens = tokenize(track.get("track_name", ""))
    reasons.extend(has_mixlike_title_flags(title_tokens))
    if has_instrumental_in_lyrics(track.get("lyrics", "")):
        reasons.append("instrumental_in_lyrics")
    if lyrics_too_short(track.get("lyrics"), cfg["min_words"], cfg["min_chars"]):
        reasons.append("lyrics_too_short")
    return list(sorted(set(reasons)))

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

def print_report(total, kept, removed, reason_counter, combo_counter, cfg, sample_removed):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(f"ResoNote Track Cleaner\nTotal: {total} | Kept: {kept} | Removed: {removed}\nDuration Window: {cfg['min_ms']/1000:.0f}s–{cfg['max_ms']/60000:.0f}m | Short Lyrics Threshold: words<{cfg['min_words']} or chars<{cfg['min_chars']}", title="Summary", border_style="cyan"))
        t1 = Table(title="Removal Reasons (Counts)", show_lines=False)
        t1.add_column("Reason", justify="left")
        t1.add_column("Count", justify="right")
        for reason, cnt in sorted(reason_counter.items(), key=lambda x: (-x[1], x[0])):
            t1.add_row(reason, str(cnt))
        console.print(t1)
        t2 = Table(title="Removal Reason Combinations", show_lines=False)
        t2.add_column("Reasons", justify="left", no_wrap=False)
        t2.add_column("Count", justify="right")
        for combo, cnt in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0])):
            t2.add_row(", ".join(combo), str(cnt))
        console.print(t2)
        if sample_removed:
            t3 = Table(title="Samples of Removed Tracks", show_lines=False)
            t3.add_column("track_id", no_wrap=True)
            t3.add_column("track_name")
            t3.add_column("reasons")
            for r in sample_removed[:20]:
                t3.add_row(str(r.get("track_id","")), str(r.get("track_name","")), ", ".join(r.get("reasons",[])))
            console.print(t3)
    except Exception:
        print(f"Total: {total} | Kept: {kept} | Removed: {removed}")
        print(f"Duration Window: {cfg['min_ms']/1000:.0f}s–{cfg['max_ms']/60000:.0f}m | Short Lyrics Threshold: words<{cfg['min_words']} or chars<{cfg['min_chars']}")
        print("\nRemoval Reasons (Counts):")
        for reason, cnt in sorted(reason_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {reason}: {cnt}")
        print("\nRemoval Reason Combinations:")
        for combo, cnt in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {', '.join(combo)}: {cnt}")
        if sample_removed:
            print("\nSamples of Removed Tracks:")
            for r in sample_removed[:20]:
                print(f"- {r.get('track_id','')} | {r.get('track_name','')} | {', '.join(r.get('reasons',[]))}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="dataset/processed/track_data/audio_features_and_lyrics.json")
    p.add_argument("--output", default="dataset/processed/track_data/audio_features_and_lyrics_cleaned.json")
    p.add_argument("--min-seconds", type=float, default=30.0)
    p.add_argument("--max-minutes", type=float, default=15.0)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--min-chars", type=int, default=40)
    args = p.parse_args()
    cfg = {
        "min_ms": int(args.min_seconds * 1000),
        "max_ms": int(args.max_minutes * 60 * 1000),
        "min_words": args.min_words,
        "min_chars": args.min_chars,
    }
    in_path = Path(args.input)
    out_path = Path(args.output)
    data = load_json_array(in_path)
    removed_info = []
    kept = []
    reason_counter = Counter()
    combo_counter = Counter()
    for tr in data:
        reasons = evaluate_reasons(tr, cfg)
        if reasons:
            removed_info.append({**{k: tr.get(k) for k in ["track_id","track_name","artist_name"]}, "reasons": reasons})
            for r in reasons:
                reason_counter[r] += 1
            combo_counter[tuple(sorted(reasons))] += 1
        else:
            kept.append(tr)
    save_json_array(out_path, kept)
    print_report(len(data), len(kept), len(removed_info), reason_counter, combo_counter, cfg, removed_info)

if __name__ == "__main__":
    main()
