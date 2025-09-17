#!/usr/bin/env python3
import json
from pathlib import Path

FACETS = ["Emotional_Tone", "Thematic_Content", "Narrative_Structure", "Lyrical_Style"]
MODEL_FILES = [
    "eval_tags_deepseek-v3p1.json",
    "eval_tags_glm-4p5.json",
    "eval_tags_llama-v3p1-8b-instruct.json",
    "eval_tags_llama-v3p1-70b-instruct.json",
    "eval_tags_llama-v3p3-70b-instruct.json",
    "eval_tags_mixtral-8x22b-instruct.json",
]

def load_json_array(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def main():
    base_dir = Path("analysis/llm_selection_p1/final_tags")
    output_dir = base_dir / "aggregates"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique = {facet: set() for facet in FACETS}
    for fname in MODEL_FILES:
        fpath = base_dir / fname
        rows = load_json_array(fpath)
        for row in rows:
            tags = row.get("tags") or {}
            if not isinstance(tags, dict):
                continue
            for facet in FACETS:
                vals = tags.get(facet)
                if not isinstance(vals, list):
                    continue
                for v in vals:
                    if isinstance(v, str) and len(v) > 0:
                        unique[facet].add(v)
    result = {facet: sorted(unique[facet], key=lambda s: s.casefold()) for facet in FACETS}
    out_path = output_dir / "all_unique_tags.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(str(out_path))

if __name__ == "__main__":
    main()
