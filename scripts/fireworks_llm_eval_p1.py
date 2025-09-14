#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fireworks import LLM
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

console = Console()
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise RuntimeError("FIREWORKS_API_KEY not set.")

FACET_KEYS = ["Emotional_Tone", "Thematic_Content", "Narrative_Structure", "Lyrical_Style"]

CANONICAL_OVERRIDES: Dict[str, str] = {
    "glm-4p5": "accounts/fireworks/models/glm-4p5",
    "fireworks/glm-4p5": "accounts/fireworks/models/glm-4p5",
    "accounts/fireworks/models/glm-4p5": "accounts/fireworks/models/glm-4p5",
    "deepseek-v3p1": "accounts/fireworks/models/deepseek-v3p1",
    "fireworks/deepseek-v3p1": "accounts/fireworks/models/deepseek-v3p1",
    "accounts/fireworks/models/deepseek-v3p1": "accounts/fireworks/models/deepseek-v3p1",
}

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

def canonicalize_model_id(model: str) -> str:
    key = model.strip()
    if key in CANONICAL_OVERRIDES:
        canon = CANONICAL_OVERRIDES[key]
        if canon != key:
            console.print(f"Using canonical id for {key}: {canon}")
        return canon
    return key

def load_tracks(p: Path) -> List[Dict[str,Any]]:
    return json.loads(p.read_text(encoding="utf-8"))

def bin_valence(x):
    if x is None:
        return "UNK"
    if x < 0.33:
        return "V0"
    if x < 0.66:
        return "V1"
    return "V2"

def bin_popularity(p):
    if p is None:
        return "UNK"
    if p < 33:
        return "P0"
    if p < 66:
        return "P1"
    return "P2"

def bin_energy(e):
    if e is None:
        return "UNK"
    if e < 0.33:
        return "E0"
    if e < 0.66:
        return "E1"
    return "E2"

def stratified_split(tracks: List[Dict[str,Any]], dev_n: int, seed: int = 13):
    rnd = random.Random(seed)
    buckets: Dict[str, List[Dict[str,Any]]] = {}
    for t in tracks:
        vb = bin_valence(t.get("valence"))
        pb = bin_popularity(t.get("popularity"))
        eb = bin_energy(t.get("energy"))
        key = f"{vb}|{pb}|{eb}"
        buckets.setdefault(key, []).append(t)
    for k in buckets:
        rnd.shuffle(buckets[k])
    target = min(dev_n, len(tracks))
    dev: List[Dict[str,Any]] = []
    seen_ids = set()
    idx_map = {k: 0 for k in buckets}
    keys = list(buckets.keys())
    ki = 0
    while len(dev) < target:
        progressed = False
        for _ in range(len(keys)):
            k = keys[ki]
            ki = (ki + 1) % len(keys)
            i = idx_map[k]
            if i >= len(buckets[k]):
                continue
            cand = buckets[k][i]
            idx_map[k] = i + 1
            ident = id(cand)
            if ident in seen_ids:
                continue
            dev.append(cand)
            seen_ids.add(ident)
            progressed = True
            if len(dev) >= target:
                break
        if not progressed:
            break
    if len(dev) < target:
        remaining = [t for t in tracks if id(t) not in seen_ids]
        rnd.shuffle(remaining)
        need = target - len(dev)
        dev.extend(remaining[:need])
        for t in remaining[:need]:
            seen_ids.add(id(t))
    evalset = [t for t in tracks if id(t) not in seen_ids]
    rnd.shuffle(dev)
    rnd.shuffle(evalset)
    return dev, evalset, buckets

def prompt_variants_p1():
    v0 = (
        "You analyze song lyrics and produce a single JSON object with exactly four keys: Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "Each key maps to a list of 1 to 5 concise tags. Use short keywords: one word preferred or hyphenated two words. If a facet is not clearly present, use the single tag Unsure for that facet. "
        "Definitions: Emotional_Tone = overarching mood or sentiment; Thematic_Content = main subjects or topics; Narrative_Structure = how the message unfolds (perspective, timeline, plot or lack thereof); Lyrical_Style = stylistic characteristics or devices. "
        "Return strictly the JSON object only, with no explanations."
    )
    v1 = (
        "Return one JSON object with keys Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "Each value is a list of 1–5 short tags (one word or hyphenated two words). Use Unsure when evidence is weak or absent. "
        "Output JSON only."
    )
    v2 = (
        "Output JSON only. Keys: Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "Each value: list of 1–5 concise tags (prefer one word or hyphenated two words). Use Unsure if facet not present."
    )
    v3 = (
        "Produce exactly one JSON object with keys Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "For each key, return 1–5 tags, each 1 word or hyphenated-2-words, lowercase if possible. Use Unsure if unclear. "
        "No prose, JSON only."
    )
    v4 = (
        "Return a single JSON object. Keys: Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "Each value is a list of 1–5 concise tags. Prefer Unsure rather than guessing when a facet is not identifiable. "
        "Return JSON only."
    )
    return {"V0_Comprehensive": v0, "V1_Compact": v1, "V2_Minimal": v2, "V3_StyleConstrained": v3, "V4_UnsureFirst": v4}

def json_schema_for_fireworks_p1():
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

def call_model_p1(llm: LLM, sys_prompt: str, lyrics: str, input_budget: int, output_budget: int, schema: Dict[str,Any], max_tokens: int, timeout: int, tokenizer, temperature: float, retries: int = 2):
    last_err = None
    for attempt in range(retries + 1):
        t0 = now_ms()
        user_content, truncated = truncate_lyrics_for_budget(lyrics or "", sys_prompt, input_budget, output_budget, tokenizer)
        try:
            resp = llm.chat.completions.create(
                messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_content}],
                response_format={"type":"json_schema","json_schema":{"name":"P1Tags","schema":schema,"strict":True}},
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                request_timeout=timeout
            )
            content = resp.choices[0].message.content
            dt = (now_ms()-t0)/1000.0
            if content is None:
                raise ValueError("empty content")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("non-object JSON")
            ok_schema = True
            for k in FACET_KEYS:
                if k not in parsed or not isinstance(parsed[k], list) or len(parsed[k]) < 1 or len(parsed[k]) > 5:
                    ok_schema = False
            inp_toks = estimate_tokens(sys_prompt + user_content, tokenizer)
            out_toks = estimate_tokens(content, tokenizer)
            return {"ok": bool(ok_schema), "parsed": parsed if ok_schema else None, "raw": content, "latency": dt, "truncated": truncated, "input_tokens_est": inp_toks, "output_tokens_est": out_toks, "error": None if ok_schema else "schema_invalid"}
        except Exception as e:
            last_err = str(e)
            time.sleep(0.7 * (attempt + 1))
    return {"ok": False, "parsed": None, "raw": None, "latency": None, "truncated": False, "input_tokens_est": None, "output_tokens_est": None, "error": last_err or "unknown_error"}

def is_style_ok(tag: str, max_len: int, max_words: int) -> bool:
    if not isinstance(tag, str):
        return False
    t = tag.strip()
    if len(t) == 0 or len(t) > max_len:
        return False
    parts = t.split()
    if len(parts) == 0 or len(parts) > max_words:
        return False
    for w in parts:
        if not re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", w):
            return False
    return True

def eval_dev_outputs(dev_results: List[Dict[str,Any]], max_tag_len: int, max_tag_words: int):
    total = len(dev_results)
    ok = 0
    latencies = []
    total_tags = 0
    style_ok = 0
    unsure = 0
    for r in dev_results:
        if r.get("ok"):
            ok += 1
            if isinstance(r.get("latency"), (int, float)):
                latencies.append(r["latency"])
            obj = r.get("parsed", {})
            for k in FACET_KEYS:
                vals = obj.get(k, [])
                for tag in vals:
                    total_tags += 1
                    if is_style_ok(str(tag), max_tag_len, max_tag_words):
                        style_ok += 1
                    if str(tag).strip().lower() == "unsure":
                        unsure += 1
    json_ok_rate = ok / total if total > 0 else 0.0
    tag_style_ok_rate = (style_ok / total_tags) if total_tags > 0 else 0.0
    unsure_rate = (unsure / total_tags) if total_tags > 0 else 0.0
    avg_tags_per_facet = (total_tags / (ok * len(FACET_KEYS))) if ok > 0 else 0.0
    p50 = statistics.median(latencies) if latencies else None
    p95 = None
    if latencies:
        arr = sorted(latencies)
        idx = max(0, math.ceil(0.95 * len(arr)) - 1)
        p95 = arr[idx]
    avg_inp = None
    avg_out = None
    inp_vals = [r["input_tokens_est"] for r in dev_results if isinstance(r.get("input_tokens_est"), int)]
    out_vals = [r["output_tokens_est"] for r in dev_results if isinstance(r.get("output_tokens_est"), int)]
    if inp_vals:
        avg_inp = sum(inp_vals) / len(inp_vals)
    if out_vals:
        avg_out = sum(out_vals) / len(out_vals)
    return {
        "json_ok_rate": round(json_ok_rate, 4),
        "tag_style_ok_rate": round(tag_style_ok_rate, 4),
        "unsure_rate": round(unsure_rate, 4),
        "avg_tags_per_facet": round(avg_tags_per_facet, 3),
        "latency_p50_s": None if p50 is None else round(p50, 3),
        "latency_p95_s": None if p95 is None else round(p95, 3),
        "input_tokens_avg": None if avg_inp is None else round(avg_inp, 1),
        "output_tokens_avg": None if avg_out is None else round(avg_out, 1),
        "total_tracks": total,
        "ok_tracks": ok
    }

def rank_and_select(results: List[Dict[str,Any]]):
    def keyf(r):
        m = r["metrics"]
        json_ok = m.get("json_ok_rate", 0.0)
        unsure = m.get("unsure_rate", 1.0)
        style_ok = m.get("tag_style_ok_rate", 0.0)
        p95 = m.get("latency_p95_s", None)
        p95v = 1e9 if p95 is None else p95
        toks = (m.get("input_tokens_avg") or 0.0) + (m.get("output_tokens_avg") or 0.0)
        return (-json_ok, unsure, -style_ok, p95v, toks)
    ranked = sorted(results, key=keyf)
    best_by_model = {}
    for r in ranked:
        m = r["model"]
        if m not in best_by_model:
            best_by_model[m] = r
    return best_by_model, ranked

def ping_serverless_model(model: str) -> Tuple[bool, str, bool]:
    try:
        llm = LLM(model=model, deployment_type="serverless", api_key=FIREWORKS_API_KEY)
        r = llm.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8,
            request_timeout=20
        )
        ch = (r.choices or [None])[0]
        if ch and getattr(ch, "message", None):
            msg = ch.message
            text = (msg.content or "").strip()
            has_text = len(text) > 0
            has_tools = bool(getattr(msg, "tool_calls", None)) or bool(getattr(msg, "function_call", None))
            ok = has_text or has_tools
            return (ok, "" if ok else "empty response", False)
        return (False, "no choices", False)
    except Exception as e:
        msg = str(e)
        if "Failed to format non-streaming choice" in msg or "Unexpected EOS" in msg:
            return True, "stream-only", True
        return False, msg, False

def build_serverless_llm_map(models: List[str]) -> Dict[str, LLM]:
    prov = Table(show_header=True, header_style="bold cyan")
    prov.add_column("Model")
    prov.add_column("Status")
    prov.add_column("Details")
    llm_map: Dict[str, LLM] = {}
    for requested in models:
        canon = canonicalize_model_id(requested)
        ok, detail, stream_only = ping_serverless_model(canon)
        if ok and not stream_only:
            llm_map[requested] = LLM(model=canon, deployment_type="serverless", api_key=FIREWORKS_API_KEY)
            prov.add_row(requested, "[green]ready[/]", "-")
        elif ok and stream_only:
            prov.add_row(requested, "[yellow]ready (stream-only)[/]", detail or "-")
        else:
            prov.add_row(requested, "[red]unavailable[/]", detail or "-")
    console.print(Panel(prov, title="Serverless Model Availability", expand=False))
    return llm_map

def run_config_on_dev(dev: List[Dict[str,Any]], llm: LLM, sys_prompt: str, input_budget: int, output_budget: int, max_workers: int, schema: Dict[str,Any], timeout: int, max_tokens: int, tokenizer: Any, max_tag_len: int, max_tag_words: int, temperature: float):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {}
        for t in dev:
            lyrics = t.get("lyrics") or ""
            fut = ex.submit(call_model_p1, llm, sys_prompt, lyrics, input_budget, output_budget, schema, max_tokens, timeout, tokenizer, temperature)
            fut_map[fut] = t
        for fut in as_completed(fut_map):
            t = fut_map[fut]
            r = fut.result()
            entry = {
                "track_id": t.get("track_id"),
                "ok": r["ok"],
                "error": r.get("error"),
                "parsed": r.get("parsed"),
                "latency": r.get("latency"),
                "truncated": r.get("truncated"),
                "input_tokens_est": r.get("input_tokens_est"),
                "output_tokens_est": r.get("output_tokens_est")
            }
            results.append(entry)
    metrics = eval_dev_outputs(results, max_tag_len, max_tag_words)
    return results, metrics

def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def run_grid(dev: List[Dict[str,Any]], llm_map: Dict[str,LLM], temps: List[float], variant_map: Dict[str,str], outdir: Path, input_budget: int, output_budget: int, max_workers: int, timeout: int, max_tokens: int, max_tag_len: int, max_tag_words: int):
    outdir.mkdir(parents=True, exist_ok=True)
    models_root = outdir / "models"
    summaries_root = outdir / "summaries"
    models_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)
    schema = json_schema_for_fireworks_p1()
    tokenizer = try_load_tokenizer()
    all_results = []
    configs = [(model, vname, temp) for model in llm_map.keys() for vname in variant_map for temp in temps]
    with Progress(SpinnerColumn(), TextColumn("[bold]Grid[/]: {task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task_configs = progress.add_task("configs", total=len(configs))
        for model, vname, temp in configs:
            llm = llm_map[model]
            sys_prompt = variant_map[vname]
            results, metrics = run_config_on_dev(dev, llm, sys_prompt, input_budget, output_budget, max_workers, schema, timeout, max_tokens, tokenizer, max_tag_len, max_tag_words, temp)
            model_dir = models_root / safe_name(model) / safe_name(vname) / f"T_{str(temp).replace('.','_')}"
            save_json(model_dir / "dev_outputs.json", results)
            entry = {"model": model, "variant": vname, "temperature": temp, "metrics": metrics}
            save_json(model_dir / "dev_metrics.json", entry)
            all_results.append(entry)
            t = Table(show_header=True, header_style="bold magenta")
            t.add_column("Model")
            t.add_column("Variant")
            t.add_column("Temp")
            t.add_column("JSON OK")
            t.add_column("Unsure")
            t.add_column("Style OK")
            t.add_column("p95(s)")
            t.add_column("Tok In/Out")
            t.add_row(
                model,
                vname,
                str(temp),
                f"{metrics['json_ok_rate']:.2f}",
                f"{metrics['unsure_rate']:.3f}",
                f"{metrics['tag_style_ok_rate']:.3f}",
                "-" if metrics["latency_p95_s"] is None else f"{metrics['latency_p95_s']:.3f}",
                f"{'-' if metrics['input_tokens_avg'] is None else int(metrics['input_tokens_avg'])}/{ '-' if metrics['output_tokens_avg'] is None else int(metrics['output_tokens_avg'])}"
            )
            console.print(Panel(t, title="Completed Config", expand=False))
            progress.update(task_configs, advance=1)
    save_json(summaries_root / "grid_summary.json", all_results)
    return all_results

def tag_full_eval(evalset: List[Dict[str,Any]], selection: Dict[str,Dict[str,Any]], llm_map: Dict[str,LLM], outdir: Path, input_budget: int, output_budget: int, max_workers: int, timeout: int, max_tokens: int):
    outdir.mkdir(parents=True, exist_ok=True)
    schema = json_schema_for_fireworks_p1()
    tokenizer = try_load_tokenizer()
    outputs = {}
    variants = prompt_variants_p1()
    with Progress(SpinnerColumn(), TextColumn("[bold]Final Tagging[/]: {task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task_models = progress.add_task("models", total=len(selection))
        for model, cfg in selection.items():
            if model not in llm_map:
                progress.update(task_models, advance=1)
                continue
            llm = llm_map[model]
            sys_prompt = variants[cfg["variant"]]
            temp = cfg["temperature"]
            final = []
            def one_call(tobj):
                r = call_model_p1(llm, sys_prompt, tobj.get("lyrics") or "", input_budget, output_budget, schema, max_tokens, timeout, tokenizer, temp)
                if r.get("ok"):
                    return {"track_id": tobj.get("track_id"), "tags": r.get("parsed")}
                else:
                    return {"track_id": tobj.get("track_id"), "tags": {"Emotional_Tone":["Unsure"],"Thematic_Content":["Unsure"],"Narrative_Structure":["Unsure"],"Lyrical_Style":["Unsure"]}}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(one_call, t) for t in evalset]
                for f in as_completed(futs):
                    final.append(f.result())
            op = outdir / f"eval_tags_{safe_name(model)}.json"
            save_json(op, final)
            outputs[model] = str(op)
            progress.update(task_models, advance=1)
    return outputs

def save_selection(best_by_model: Dict[str,Any], ranked: List[Dict[str,Any]], outdir: Path):
    summaries_root = outdir / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)
    sel = {"best_by_model": best_by_model, "ranked": ranked}
    save_json(summaries_root / "llm_selection_results.json", sel)

def combo_key(t):
    return f"{bin_valence(t.get('valence'))}|{bin_popularity(t.get('popularity'))}|{bin_energy(t.get('energy'))}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_size", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0,0.2,0.5])
    ap.add_argument("--models", type=str, nargs="+", default=[
        "llama-v3p3-70b-instruct",
        "llama-v3p1-70b-instruct",
        "mixtral-8x22b-instruct",
        "llama-v3p1-8b-instruct",
        "glm-4p5",
        "deepseek-v3p1"
    ])
    ap.add_argument("--outdir", type=str, default="analysis/llm_selection_p1")
    ap.add_argument("--input_token_budget", type=int, default=3500)
    ap.add_argument("--output_token_budget", type=int, default=256)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_tag_len", type=int, default=24)
    ap.add_argument("--max_tag_words", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)
    t0 = time.time()
    input_path = Path("dataset/stratified/250_sample_tracks.json")
    tracks = load_tracks(input_path)
    variants = prompt_variants_p1()
    dev, evalset, buckets = stratified_split(tracks, args.dev_size, seed=args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir,"splits").mkdir(parents=True, exist_ok=True)
    save_json(Path(args.outdir,"splits","dev.json"), dev)
    save_json(Path(args.outdir,"splits","eval.json"), evalset)
    combos = Counter(combo_key(t) for t in tracks)
    console.print(Panel(str(combos.most_common(10)), title="Top Stratification Buckets", expand=False))
    console.rule("[bold blue]Fireworks LLM Selection P1")
    info = Table(show_header=False, box=None)
    info.add_row("Input", str(input_path))
    info.add_row("Total Tracks", str(len(tracks)))
    info.add_row("Buckets", str(len(buckets)))
    info.add_row("Dev Size", str(len(dev)))
    info.add_row("Eval Size", str(len(evalset)))
    info.add_row("Max Workers", str(args.max_workers))
    info.add_row("Temps", ", ".join(map(str, args.temps)))
    info.add_row("Models", ", ".join(args.models))
    info.add_row("Input Token Budget", str(args.input_token_budget))
    info.add_row("Output Token Budget", str(args.output_token_budget))
    console.print(Panel(info, title="Run Configuration", expand=False))
    llm_map = build_serverless_llm_map(args.models)
    if not llm_map:
        console.print("[red]No serverless models are available. Exiting.[/]")
        return
    console.print("[bold]Starting grid search[/]...")
    results = run_grid(dev, llm_map, args.temps, variants, Path(args.outdir), args.input_token_budget, args.output_token_budget, args.max_workers, args.timeout_s, args.max_tokens, args.max_tag_len, args.max_tag_words)
    best_by_model, ranked = rank_and_select(results)
    save_selection(best_by_model, ranked, Path(args.outdir))
    rt = Table(show_header=True, header_style="bold magenta")
    rt.add_column("Rank")
    rt.add_column("Model")
    rt.add_column("Variant")
    rt.add_column("Temp")
    rt.add_column("JSON OK")
    rt.add_column("Unsure")
    rt.add_column("Style OK")
    rt.add_column("p95(s)")
    for i, r in enumerate(ranked[:10], start=1):
        m = r["metrics"]
        rt.add_row(
            str(i),
            r["model"],
            r["variant"],
            str(r["temperature"]),
            f"{m['json_ok_rate']:.2f}",
            f"{m['unsure_rate']:.3f}",
            f"{m['tag_style_ok_rate']:.3f}",
            "-" if m['latency_p95_s'] is None else f"{m['latency_p95_s']:.3f}"
        )
    console.print(Panel(rt, title="Top Configs", expand=False))
    console.print("[bold]Final tagging on eval set[/]...")
    tag_outputs = tag_full_eval(evalset, best_by_model, llm_map, Path(args.outdir,"final_tags"), args.input_token_budget, args.output_token_budget, args.max_workers, args.timeout_s, args.max_tokens)
    elapsed = time.time() - t0
    summary = Table(show_header=False, box=None)
    summary.add_row("Elapsed", f"{elapsed:.1f}s")
    summary.add_row("Selection File", str(Path(args.outdir,"summaries","llm_selection_results.json")))
    summary.add_row("Grid Summary", str(Path(args.outdir,"summaries","grid_summary.json")))
    summary.add_row("Final Tags Dir", str(Path(args.outdir,"final_tags")))
    console.print(Panel(summary, title="Done", expand=False))
    print(json.dumps({"best_by_model": best_by_model, "outputs": tag_outputs}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
