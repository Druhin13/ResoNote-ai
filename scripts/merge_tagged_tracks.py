#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List, Generator, Any


def load_json_or_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        v["_key"] = k
                        yield v
                    else:
                        yield {"_key": k, "value": v}
            elif isinstance(data, list):
                for item in data:
                    yield item
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def merge_tagged_tracks(file_paths: List[Path], output_path: Path, key: str, filter_unmatched: bool):
    merged_dict: Dict[str, Dict[str, Any]] = {}
    file_sources = []

    for path in file_paths:
        source_name = path.name
        file_sources.append(source_name)

        for item in load_json_or_jsonl(path):
            item_key = item.get(key) or item.get("_key")
            if not item_key:
                continue

            item_key = str(item_key)

            if item_key not in merged_dict:
                merged_dict[item_key] = {}

            merged_dict[item_key].update(item)

    with output_path.open("w", encoding="utf-8") as out_f:
        for k, record in merged_dict.items():
            if filter_unmatched and any(record.get(key) is None for path in file_paths):
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple JSON/JSONL files by a shared key.")
    parser.add_argument("inputs", nargs="+", help="Input files (.json or .jsonl), at least 2")
    parser.add_argument("--out", required=True, help="Path to output .jsonl file")
    parser.add_argument("--key", default="track_id", help="Field to merge on (default: track_id)")
    parser.add_argument("--filter-unmatched", action="store_true", help="Exclude records that don't appear in all input files")

    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("At least 2 input files are required.")

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.out)

    merge_tagged_tracks(
        file_paths=input_paths,
        output_path=output_path,
        key=args.key,
        filter_unmatched=args.filter_unmatched
    )


if __name__ == "__main__":
    main()
