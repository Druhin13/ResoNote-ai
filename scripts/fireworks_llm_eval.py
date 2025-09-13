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

ALLOWED_TAGS = {
    "Emotional_Tone": ["Joy","Sadness","Anger","Bittersweet","Nostalgia","Euphoria","Triumph","Regret","Loneliness","None"],
    "Thematic_Content": ["Love","Heartbreak","Wealth","Success","Struggle","Party","Protest","Spirituality","None"],
    "Narrative_Structure": ["Narrative_Yes","Narrative_No","Conflict_Resolution","First_Person","Third_Person","None"],
    "Lyrical_Style": ["Repetition","Metaphor","Alliteration","Vivid_Imagery","Rhetorical_Question","None"]
}

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

def prompt_variants():
    tag_lists = {
        "Emotional_Tone": ",".join(f'"{x}"' for x in ALLOWED_TAGS["Emotional_Tone"]),
        "Thematic_Content": ",".join(f'"{x}"' for x in ALLOWED_TAGS["Thematic_Content"]),
        "Narrative_Structure": ",".join(f'"{x}"' for x in ALLOWED_TAGS["Narrative_Structure"]),
        "Lyrical_Style": ",".join(f'"{x}"' for x in ALLOWED_TAGS["Lyrical_Style"])
    }
    schema_text = (
        'Return an array of objects where each element is {"track_id": "string","tags":{"Emotional_Tone":[{"tag":"string","score":0.0}],"Thematic_Content":[{"tag":"string","score":0.0}],"Narrative_Structure":[{"tag":"string","score":0.0}],"Lyrical_Style":[{"tag":"string","score":0.0}]}}. '
        'Use only the allowed tag lists. If a facet is not clearly expressed, use "None" with score 1.0. Reply with strictly valid JSON and nothing else.'
    )
    allowed_block = (
        f'Allowed tags:\n- Emotional_Tone: [{tag_lists["Emotional_Tone"]}]\n'
        f'- Thematic_Content: [{tag_lists["Thematic_Content"]}]\n'
        f'- Narrative_Structure: [{tag_lists["Narrative_Structure"]}]\n'
        f'- Lyrical_Style: [{tag_lists["Lyrical_Style"]}]'
    )
    v0 = "You are a music semantic tagger. " + schema_text + "\n" + allowed_block
    v1 = "You are a music semantic tagging assistant. Follow the schema precisely. Do not invent fields. " + schema_text + "\n" + allowed_block
    v2 = "Rules: Only allowed tags. No extra fields. One JSON array as final output. " + schema_text + "\n" + allowed_block
    v3 = "Tag conservatively. Prefer 'None' when evidence is weak. " + schema_text + "\n" + allowed_block
    v4 = "Abstain with 'None' when lyrics are minimalistic, abstract, or contradictory. " + schema_text + "\n" + allowed_block
    return {"V0_MinSchema": v0, "V1_Baseline": v1, "V2_ConstraintFirst": v2, "V3_EvidenceFirst": v3, "V4_RobustNone": v4}

def json_schema_for_fireworks():
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "track_id": {"type": "string"},
                "tags": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Emotional_Tone": {
                            "type":"array",
                            "items": {
                                "type":"object",
                                "additionalProperties": False,
                                "properties":{
                                    "tag":{"type":"string","enum":ALLOWED_TAGS["Emotional_Tone"]},
                                    "score":{"type":"number"}
                                },
                                "required":["tag","score"]
                            },
                            "minItems":1
                        },
                        "Thematic_Content": {
                            "type":"array",
                            "items": {
                                "type":"object",
                                "additionalProperties": False,
                                "properties":{
                                    "tag":{"type":"string","enum":ALLOWED_TAGS["Thematic_Content"]},
                                    "score":{"type":"number"}
                                },
                                "required":["tag","score"]
                            },
                            "minItems":1
                        },
                        "Narrative_Structure": {
                            "type":"array",
                            "items": {
                                "type":"object",
                                "additionalProperties": False,
                                "properties":{
                                    "tag":{"type":"string","enum":ALLOWED_TAGS["Narrative_Structure"]},
                                    "score":{"type":"number"}
                                },
                                "required":["tag","score"]
                            },
                            "minItems":1
                        },
                        "Lyrical_Style": {
                            "type":"array",
                            "items": {
                                "type":"object",
                                "additionalProperties": False,
                                "properties":{
                                    "tag":{"type":"string","enum":ALLOWED_TAGS["Lyrical_Style"]},
                                    "score":{"type":"number"}
                                },
                                "required":["tag","score"]
                            },
                            "minItems":1
                        }
                    },
                    "required": ["Emotional_Tone","Thematic_Content","Narrative_Structure","Lyrical_Style"]
                }
            },
            "required": ["track_id","tags"]
        }
    }

def build_user_batch(batch: List[Dict[str,Any]]) -> str:
    payload = []
    for t in batch:
        payload.append({
            "track_id": t.get("track_id"),
            "track_name": t.get("track_name"),
            "artist_name": t.get("artist_name"),
            "audio_features": {
                "acousticness": t.get("acousticness"),
                "danceability": t.get("danceability"),
                "duration_ms": t.get("duration_ms"),
                "energy": t.get("energy"),
                "instrumentalness": t.get("instrumentalness"),
                "key": t.get("key"),
                "liveness": t.get("liveness"),
                "loudness": t.get("loudness"),
                "mode": t.get("mode"),
                "speechiness": t.get("speechiness"),
                "tempo": t.get("tempo"),
                "time_signature": t.get("time_signature"),
                "valence": t.get("valence"),
                "popularity": t.get("popularity")
            },
            "lyrics": t.get("lyrics") or ""
        })
    return json.dumps(payload, ensure_ascii=False)

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def now_ms():
    return int(time.time()*1000)

def call_model(llm: LLM, sys_prompt: str, user_json: str, temperature: float, schema: Dict[str,Any], max_tokens: int, timeout: int):
    t0 = now_ms()
    try:
        resp = llm.chat.completions.create(
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_json}],
            response_format={"type":"json_schema","json_schema":{"name":"TrackTags","schema":schema,"strict":True}},
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            request_timeout=timeout
        )
        content = resp.choices[0].message.content
        dt = (now_ms()-t0)/1000.0
        if content is None:
            raise ValueError("LLM returned empty content")
        parsed = json.loads(content)
        return {"ok":True,"latency":dt,"parsed":parsed,"raw":content}
    except Exception as e:
        dt = (now_ms()-t0)/1000.0
        return {"ok":False,"latency":dt,"error":str(e)}

def eval_batch(parsed: Any) -> Dict[str,Any]:
    try:
        arr = list(parsed)
    except Exception:
        return {"json_valid":False,"oov":0,"total_tags":0,"none_counts":{"Emotional_Tone":0,"Thematic_Content":0,"Narrative_Structure":0,"Lyrical_Style":0},"cardinality_err":1}
    json_valid = True
    oov = 0
    total = 0
    cardinality_err = 0
    none_counts = {"Emotional_Tone":0,"Thematic_Content":0,"Narrative_Structure":0,"Lyrical_Style":0}
    for obj in arr:
        tags = obj.get("tags",{})
        for facet in ALLOWED_TAGS.keys():
            vals = tags.get(facet,[])
            if not isinstance(vals,list) or len(vals)==0:
                cardinality_err += 1
                continue
            for it in vals:
                tag = it.get("tag")
                score = it.get("score")
                if tag not in ALLOWED_TAGS[facet]:
                    oov += 1
                if tag == "None":
                    none_counts[facet] += 1
                if not isinstance(score,(int,float)):
                    cardinality_err += 1
                total += 1
    return {"json_valid":json_valid,"oov":oov,"total_tags":total,"none_counts":none_counts,"cardinality_err":cardinality_err}

def aggregate_metrics(batch_metrics: List[Dict[str,Any]], batch_latencies: List[float], batches_ok: int, batches_total: int):
    json_validity_rate = batches_ok / max(1,batches_total)
    oov = sum(m["oov"] for m in batch_metrics)
    total_tags = sum(m["total_tags"] for m in batch_metrics)
    oov_rate = (oov / total_tags) if total_tags>0 else 0.0
    none_totals = {"Emotional_Tone":0,"Thematic_Content":0,"Narrative_Structure":0,"Lyrical_Style":0}
    for m in batch_metrics:
        for k in none_totals:
            none_totals[k] += m["none_counts"][k]
    card_err = sum(m["cardinality_err"] for m in batch_metrics)
    p50 = statistics.median(batch_latencies) if batch_latencies else None
    p95 = None
    if batch_latencies:
        arr = sorted(batch_latencies)
        idx = math.ceil(0.95*len(arr))-1
        p95 = arr[max(0,idx)]
    return {
        "json_validity_rate": round(json_validity_rate,4),
        "oov_rate": round(oov_rate,4),
        "cardinality_errors": card_err,
        "latency_p50_s": None if p50 is None else round(p50,3),
        "latency_p95_s": None if p95 is None else round(p95,3),
        "total_tags": total_tags,
        "none_counts": none_totals
    }

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

def run_grid(tracks: List[Dict[str,Any]], llm_map: Dict[str,LLM], temps: List[float], variant_map: Dict[str,str], batch_size: int, max_workers: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    models_root = outdir / "models"
    summaries_root = outdir / "summaries"
    models_root.mkdir(parents=True, exist_ok=True)
    summaries_root.mkdir(parents=True, exist_ok=True)
    schema = json_schema_for_fireworks()
    results = []
    configs = [(model, vname, temp) for model in llm_map.keys() for vname in variant_map for temp in temps]
    with Progress(SpinnerColumn(), TextColumn("[bold]Grid[/]: {task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task_configs = progress.add_task("configs", total=len(configs))
        for model, vname, temp in configs:
            vprompt = variant_map[vname]
            llm = llm_map[model]
            batches = list(chunk(tracks, batch_size))
            task_batches = progress.add_task(f"{model} | {vname} | T={temp}", total=len(batches))
            batch_metrics = []
            batch_lat = []
            ok_batches = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(call_model, llm, vprompt, build_user_batch(b), temp, schema, 2048, 120) for b in batches]
                for f in as_completed(futs):
                    r = f.result()
                    if r["ok"]:
                        ok_batches += 1
                        m = eval_batch(r["parsed"])
                        batch_metrics.append(m)
                    else:
                        batch_metrics.append({"json_valid":False,"oov":0,"total_tags":0,"none_counts":{"Emotional_Tone":0,"Thematic_Content":0,"Narrative_Structure":0,"Lyrical_Style":0},"cardinality_err":1})
                    batch_lat.append(r["latency"])
                    progress.update(task_batches, advance=1)
            agg = aggregate_metrics(batch_metrics, batch_lat, ok_batches, len(batches))
            entry = {"model":model,"variant":vname,"temperature":temp,"metrics":agg}
            results.append(entry)
            model_dir = models_root / safe_name(model)
            variant_dir = model_dir / safe_name(vname)
            model_dir.mkdir(parents=True, exist_ok=True)
            variant_dir.mkdir(parents=True, exist_ok=True)
            temp_str = str(temp).replace(".","_")
            path = variant_dir / f"T_{temp_str}.json"
            path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            t = Table(show_header=True, header_style="bold magenta")
            t.add_column("Model")
            t.add_column("Variant")
            t.add_column("Temp")
            t.add_column("JSON OK")
            t.add_column("OOV Rate")
            t.add_column("Card Err")
            t.add_column("p50(s)")
            t.add_column("p95(s)")
            t.add_row(model, vname, str(temp), f"{agg['json_validity_rate']:.2f}", f"{agg['oov_rate']:.3f}", str(agg["cardinality_errors"]), "-" if agg["latency_p50_s"] is None else f"{agg['latency_p50_s']:.3f}", "-" if agg['latency_p95_s'] is None else f"{agg['latency_p95_s']:.3f}")
            console.print(Panel(t, title="Completed Config", expand=False))
            progress.update(task_configs, advance=1)
    grid_path = summaries_root / "grid_summary.json"
    grid_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

def rank_and_select(results: List[Dict[str,Any]]):
    ranked = sorted(results, key=lambda x: (-x["metrics"]["json_validity_rate"], x["metrics"]["oov_rate"], x["metrics"]["latency_p95_s"] if x["metrics"]["latency_p95_s"] is not None else 1e9))
    best_by_model = {}
    for r in ranked:
        m = r["model"]
        if m not in best_by_model:
            best_by_model[m] = r
    return best_by_model, ranked

def tag_full_eval(tracks: List[Dict[str,Any]], selection: Dict[str,Dict[str,Any]], llm_map: Dict[str,LLM], batch_size: int, max_workers: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    schema = json_schema_for_fireworks()
    outputs = {}
    with Progress(SpinnerColumn(), TextColumn("[bold]Final Tagging[/]: {task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task_models = progress.add_task("models", total=len(selection))
        for model, cfg in selection.items():
            if model not in llm_map:
                continue
            llm = llm_map[model]
            variant = cfg["variant"]
            temp = cfg["temperature"]
            vprompt = prompt_variants()[variant]
            batches = list(chunk(tracks, batch_size))
            task_batches = progress.add_task(f"{model} | {variant} | T={temp}", total=len(batches))
            all_tagged = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(call_model, llm, vprompt, build_user_batch(b), temp, schema, 2048, 180) for b in batches]
                for f in as_completed(futs):
                    r = f.result()
                    if r["ok"]:
                        try:
                            arr = json.loads(r["raw"])
                        except Exception:
                            arr = r["parsed"]
                        for item in arr:
                            all_tagged.append(item)
                    progress.update(task_batches, advance=1)
            op = outdir / f"eval_tags_{safe_name(model)}_.json"
            op.write_text(json.dumps(all_tagged, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs[model] = str(op)
            progress.update(task_models, advance=1)
    return outputs

def save_selection(best_by_model: Dict[str,Any], ranked: List[Dict[str,Any]], outdir: Path):
    summaries_root = outdir / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)
    sel = {"best_by_model": best_by_model, "ranked": ranked}
    (summaries_root / "llm_selection_results.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")

def combo_key(t):
    return f"{bin_valence(t.get('valence'))}|{bin_popularity(t.get('popularity'))}|{bin_energy(t.get('energy'))}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--dev_size", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=5)
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
    ap.add_argument("--outdir", type=str, default="analysis/llm_selection")
    args = ap.parse_args()
    t0 = time.time()
    tracks = load_tracks(Path(args.input))
    combos = Counter(combo_key(t) for t in tracks)
    console.print(Panel(str(combos.most_common(10)), title="Top Stratification Buckets", expand=False))
    dev, evalset, buckets = stratified_split(tracks, args.dev_size)
    console.rule("[bold blue]Fireworks LLM Selection")
    info = Table(show_header=False, box=None)
    info.add_row("Input", str(args.input))
    info.add_row("Total Tracks", str(len(tracks)))
    info.add_row("Buckets", str(len(buckets)))
    info.add_row("Dev Size", str(len(dev)))
    info.add_row("Eval Size", str(len(evalset)))
    info.add_row("Batch Size", str(args.batch_size))
    info.add_row("Max Workers", str(args.max_workers))
    info.add_row("Temps", ", ".join(map(str, args.temps)))
    info.add_row("Models", ", ".join(args.models))
    console.print(Panel(info, title="Run Configuration", expand=False))
    variants = prompt_variants()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir,"splits").mkdir(parents=True, exist_ok=True)
    Path(args.outdir,"splits","dev.json").write_text(json.dumps(dev, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.outdir,"splits","eval.json").write_text(json.dumps(evalset, indent=2, ensure_ascii=False), encoding="utf-8")
    llm_map = build_serverless_llm_map(args.models)
    if not llm_map:
        console.print("[red]No serverless models are available. Exiting.[/]")
        return
    console.print("[bold]Starting grid search[/]...")
    results = run_grid(dev, llm_map, args.temps, variants, args.batch_size, args.max_workers, Path(args.outdir))
    best_by_model, ranked = rank_and_select(results)
    save_selection(best_by_model, ranked, Path(args.outdir))
    rt = Table(show_header=True, header_style="bold magenta")
    rt.add_column("Rank")
    rt.add_column("Model")
    rt.add_column("Variant")
    rt.add_column("Temp")
    rt.add_column("JSON OK")
    rt.add_column("OOV Rate")
    rt.add_column("p95(s)")
    for i, r in enumerate(ranked[:10], start=1):
        m = r["metrics"]
        rt.add_row(str(i), r["model"], r["variant"], str(r["temperature"]), f"{m['json_validity_rate']:.2f}", f"{m['oov_rate']:.3f}", "-" if m['latency_p95_s'] is None else f"{m['latency_p95_s']:.3f}")
    console.print(Panel(rt, title="Top Configs", expand=False))
    console.print("[bold]Final tagging on eval set[/]...")
    tag_outputs = tag_full_eval(evalset, best_by_model, llm_map, args.batch_size, args.max_workers, Path(args.outdir,"final_tags"))
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
