#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from fireworks import LLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
from dotenv import load_dotenv

FACETS = ["Emotional_Tone","Thematic_Content","Narrative_Structure","Lyrical_Style"]
console = Console()

def load_env(env_path: str | None):
    if env_path and Path(env_path).exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

def load_json_or_jsonl(path: Path) -> List[Dict[str,Any]]:
    text = path.read_text(encoding="utf-8")
    if text.strip().startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

def load_eval_exclusions(path: Path) -> set:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = set()
    for rec in data:
        tid = rec.get("track_id")
        if tid:
            ids.add(str(tid))
    return ids

def try_load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def estimate_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))

def truncate_lyrics_for_budget(lyrics: str, sys_prompt: str, input_budget: int, output_budget: int, tokenizer) -> Tuple[str, bool]:
    if lyrics is None:
        lyrics = ""
    header = "Lyrics:\n"
    base = sys_prompt
    total_budget = max(0, input_budget - output_budget)
    chunks = lyrics.splitlines()
    lo, hi = 0, len(chunks)
    best = ""
    truncated = False
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = "\n".join(chunks[:mid])
        content = header + candidate
        toks = estimate_tokens(base + content, tokenizer)
        if toks <= total_budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    used = header + best
    if best != "\n".join(chunks):
        used = used + "\n… [truncated for context]\n"
        truncated = True
    return used, truncated

def now_ms():
    return int(time.time()*1000)

def json_schema_for_tags():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Emotional_Tone": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "Thematic_Content": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "Narrative_Structure": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "Lyrical_Style": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}
        },
        "required": ["Emotional_Tone","Thematic_Content","Narrative_Structure","Lyrical_Style"]
    }

def v0_comprehensive_prompt():
    return (
        "You analyze song lyrics and produce a single JSON object with exactly four keys: Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "Each key maps to a list of 1 to 5 concise tags. Use short keywords: one word preferred or hyphenated two words. If a facet is not clearly present, use the single tag Unsure for that facet. "
        "Definitions: Emotional_Tone = overarching mood or sentiment; Thematic_Content = main subjects or topics; Narrative_Structure = how the message unfolds (perspective, timeline, plot or lack thereof); Lyrical_Style = stylistic characteristics or devices. "
        "Return strictly the JSON object only, with no explanations."
    )

def call_model(llm: LLM, sys_prompt: str, lyrics: str, input_budget: int, output_budget: int, schema: Dict[str,Any], max_tokens: int, timeout: int, tokenizer, temperature: float):
    t0 = now_ms()
    user_content, truncated = truncate_lyrics_for_budget(lyrics or "", sys_prompt, input_budget, output_budget, tokenizer)
    resp = llm.chat.completions.create(
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_content}],
        response_format={"type":"json_schema","json_schema":{"name":"P1Tags","schema":schema,"strict":True}},
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        request_timeout=timeout
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("empty content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("non-object JSON")
    for k in FACETS:
        if k not in parsed or not isinstance(parsed[k], list) or len(parsed[k]) < 1 or len(parsed[k]) > 5:
            raise ValueError("schema_invalid")
    dt = (now_ms()-t0)/1000.0
    inp_toks = estimate_tokens(sys_prompt + user_content, tokenizer)
    out_toks = estimate_tokens(content, tokenizer)
    return parsed, dt, truncated, inp_toks, out_toks

def fallback_unsure():
    return {"Emotional_Tone":["Unsure"],"Thematic_Content":["Unsure"],"Narrative_Structure":["Unsure"],"Lyrical_Style":["Unsure"]}

def build_llm(model_id: str) -> LLM:
    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY not set.")
    return LLM(model=model_id, deployment_type="serverless", api_key=key)

def tag_tracks(tracks: List[Dict[str,Any]], model_id: str, temperature: float, input_budget: int, output_budget: int, max_tokens: int, timeout_s: int, max_workers: int, out_jsonl: Path, manifest_path: Path):
    llm = build_llm(model_id)
    schema = json_schema_for_tags()
    tokenizer = try_load_tokenizer()
    sys_prompt = v0_comprehensive_prompt()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    in_tokens = []
    out_tokens = []
    truncated_ct = 0
    def one(t):
        tid = t.get("track_id")
        lyr = t.get("lyrics") or ""
        try:
            parsed, dt, trunc, it, ot = call_model(llm, sys_prompt, lyr, input_budget, output_budget, schema, max_tokens, timeout_s, tokenizer, temperature)
            return {"track_id": tid, "tags": parsed, "ok": True, "latency": dt, "truncated": trunc, "input_tokens_est": it, "output_tokens_est": ot}
        except Exception:
            return {"track_id": tid, "tags": fallback_unsure(), "ok": False, "latency": None, "truncated": False, "input_tokens_est": None, "output_tokens_est": None}
    with out_jsonl.open("w", encoding="utf-8") as fo, Progress(SpinnerColumn(), TextColumn("[bold]Tagging[/]"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("tracks", total=len(tracks))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(one, t) for t in tracks]
            for fut in as_completed(futs):
                rec = fut.result()
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["ok"] and isinstance(rec["latency"], (int,float)):
                    latencies.append(rec["latency"])
                if rec["ok"] and isinstance(rec["input_tokens_est"], int):
                    in_tokens.append(rec["input_tokens_est"])
                if rec["ok"] and isinstance(rec["output_tokens_est"], int):
                    out_tokens.append(rec["output_tokens_est"])
                if rec.get("truncated"):
                    truncated_ct += 1
                progress.update(task, advance=1)
    m = {
        "model": model_id,
        "variant": "V0_Comprehensive",
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "input_token_budget": input_budget,
        "output_token_budget": output_budget,
        "timeout_s": timeout_s,
        "total_input_tracks": len(tracks),
        "outputs_jsonl": str(out_jsonl),
        "truncated_contexts": truncated_ct,
        "latency_avg_s": None if not latencies else round(sum(latencies)/len(latencies), 3),
        "input_tokens_avg": None if not in_tokens else round(sum(in_tokens)/len(in_tokens), 1),
        "output_tokens_avg": None if not out_tokens else round(sum(out_tokens)/len(out_tokens), 1)
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/processed/track_data/audio_features_and_lyrics_cleaned.json")
    ap.add_argument("--exclude_eval", default="analysis/llm_selection_p1/final_tags/eval_tags_deepseek-v3p1.json")
    ap.add_argument("--output_dir", default="dataset/tagged/deepseek-v3p1")
    ap.add_argument("--model_id", default="accounts/fireworks/models/deepseek-v3p1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--input_token_budget", type=int, default=3500)
    ap.add_argument("--output_token_budget", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env_path", type=str, default="")
    args = ap.parse_args()
    load_env(args.env_path if args.env_path else None)
    ds_path = Path(args.dataset)
    ex_path = Path(args.exclude_eval)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tracks = load_json_or_jsonl(ds_path)
    excluded = load_eval_exclusions(ex_path)
    eligible = [t for t in all_tracks if str(t.get("track_id")) not in excluded]
    if args.limit and args.limit > 0:
        eligible = eligible[:args.limit]
    hdr = Table(show_header=True, header_style="bold cyan")
    hdr.add_column("Dataset")
    hdr.add_column("Total")
    hdr.add_column("Excluded")
    hdr.add_column("Eligible")
    hdr.add_row(str(ds_path), str(len(all_tracks)), str(len(excluded)), str(len(eligible)))
    console.print(Panel(hdr, title="Input Summary", expand=False))
    if len(eligible) == 0:
        console.print("[yellow]No eligible tracks to tag.[/]")
        return
    out_jsonl = out_dir / (f"sample_{args.limit}.jsonl" if args.limit and args.limit > 0 else "full_run.jsonl")
    manifest = out_dir / "run_manifest.json"
    res = tag_tracks(eligible, args.model_id, args.temperature, args.input_token_budget, args.output_token_budget, args.max_tokens, args.timeout_s, args.max_workers, out_jsonl, manifest)
    summ = Table(show_header=False, box=None)
    summ.add_row("Outputs", str(out_jsonl))
    summ.add_row("Manifest", str(manifest))
    summ.add_row("Tracks Tagged", str(len(eligible)))
    summ.add_row("Avg Latency (s)", "-" if res["latency_avg_s"] is None else str(res["latency_avg_s"]))
    summ.add_row("Avg Input Tokens", "-" if res["input_tokens_avg"] is None else str(res["input_tokens_avg"]))
    summ.add_row("Avg Output Tokens", "-" if res["output_tokens_avg"] is None else str(res["output_tokens_avg"]))
    console.print(Panel(summ, title="Run Summary", expand=False))

if __name__ == "__main__":
    main()
