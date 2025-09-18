#!/usr/bin/env python3
import json
import re
import argparse
import unicodedata
from pathlib import Path

FACETS = ("Emotional_Tone","Thematic_Content","Narrative_Structure","Lyrical_Style")
WS = re.compile(r"\s+")
DASHES = re.compile(r"-{2,}")
ALNUM_HYPHEN = re.compile(r"[^a-z0-9-]+")

def normalize_basic(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = s.replace("_", " ")
    s = WS.sub("-", s)
    s = DASHES.sub("-", s)
    s = s.strip("-")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = ALNUM_HYPHEN.sub("", s)
    s = DASHES.sub("-", s).strip("-")
    return s

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("dataset/tagged/deepseek-v3p1/all_tracks.jsonl"))
    ap.add_argument("--dst", type=Path, default=Path("analysis/llm_tagging/all_tracks_unique_tags.json"))
    ap.add_argument("--no_normalize", action="store_true")
    args = ap.parse_args()
    per_facet = {f: [] for f in FACETS}
    seen = {f: set() for f in FACETS}
    for rec in load_jsonl(args.src):
        tags = rec.get("tags") or {}
        for facet in FACETS:
            vals = tags.get(facet) or []
            for v in vals:
                if not isinstance(v, str):
                    continue
                t = v if args.no_normalize else normalize_basic(v)
                if not t:
                    continue
                if t not in seen[facet]:
                    seen[facet].add(t)
                    per_facet[facet].append(t)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8") as f:
        json.dump(per_facet, f, ensure_ascii=False, indent=2)
    print(str(args.dst))

if __name__ == "__main__":
    main()
