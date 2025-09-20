#!/usr/bin/env python3
import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fireworks import LLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

FACETS = ["Emotional_Tone","Thematic_Content","Narrative_Structure","Lyrical_Style"]
console = Console()

def load_env(env_path: Optional[str]):
    if env_path and Path(env_path).exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

def load_tracks(path: Path) -> List[Dict[str,Any]]:
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

def load_canonical(path: Path) -> Dict[str,List[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    canon = {}
    for f in FACETS:
        vals = obj.get(f, [])
        canon[f] = sorted({str(x).strip() for x in vals if isinstance(x, str) and str(x).strip()})
    return canon

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
    total_budget = max(0, input_budget - output_budget)
    chunks = lyrics.splitlines()
    lo, hi = 0, len(chunks)
    best = ""
    truncated = False
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = "\n".join(chunks[:mid])
        toks = estimate_tokens(sys_prompt + header + candidate, tokenizer)
        if toks <= total_budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    used = header + best
    if best != "\n".join(chunks):
        used = used + "\n… [truncated]\n"
        truncated = True
    return used, truncated

def now_ms():
    return int(time.time()*1000)

def build_schema_strict_full(canon: Dict[str,List[str]]) -> Dict[str,Any]:
    props = {}
    for f in FACETS:
        props[f] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tag": {"type": "string", "enum": canon[f]},
                    "p": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["tag","p"]
            },
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True
        }
    return {"type": "object","additionalProperties": False,"properties": props,"required": FACETS}

def build_schema_relaxed_full() -> Dict[str,Any]:
    props = {}
    for f in FACETS:
        props[f] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"tag": {"type": "string"}, "p": {"type": "number"}},
                "required": ["tag","p"]
            },
            "minItems": 0,
            "maxItems": 8,
            "uniqueItems": False
        }
    return {"type": "object","additionalProperties": False,"properties": props,"required": FACETS}

def build_schema_strict_facet(canon: Dict[str,List[str]], facet: str) -> Dict[str,Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"tag": {"type": "string", "enum": canon[facet]}, "p": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
            "required": ["tag","p"]
        },
        "minItems": 1,
        "maxItems": 5,
        "uniqueItems": True
    }

def join_tags(tags: List[str]) -> str:
    return ", ".join(tags)

def build_prompt_strict_full(canon: Dict[str,List[str]]) -> str:
    return (
        "You are an expert music annotator. Return exactly one JSON object with keys Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "For each key, output 1–5 items. Each item must be {\"tag\": \"<allowed>\", \"p\": <0..1>}. Use only allowed tags below. Prefer the smallest accurate set. Output JSON only.\n\n"
        "Emotional_Tone: " + join_tags(canon["Emotional_Tone"]) + "\n"
        "Thematic_Content: " + join_tags(canon["Thematic_Content"]) + "\n"
        "Narrative_Structure: " + join_tags(canon["Narrative_Structure"]) + "\n"
        "Lyrical_Style: " + join_tags(canon["Lyrical_Style"])
    )

def build_prompt_relaxed_full() -> str:
    return (
        "You are an expert music annotator. Return exactly one JSON object with keys Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style. "
        "For each key, propose 1–8 items of the form {\"tag\":\"<string>\",\"p\":<0..1>}. Prefer fewer items rather than guessing. Output JSON only."
    )

def build_prompt_strict_facet(canon: Dict[str,List[str]], facet: str) -> str:
    return (
        "You are an expert music annotator. For the facet \"" + facet + "\" only, return exactly one JSON array with 1–5 objects of the form {\"tag\":\"<allowed>\",\"p\":<0..1>}. "
        "Use only allowed tags: " + join_tags(canon[facet]) + ". Output JSON array only."
    )

def build_llm(model_id: str) -> LLM:
    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY not set.")
    return LLM(model=model_id, deployment_type="serverless", api_key=key)

def call_model(llm: LLM, sys_prompt: str, lyrics: str, input_budget: int, output_budget: int, schema: Dict[str,Any], max_tokens: int, timeout: int, tokenizer, temperature: float, retries: int = 1, backoff: float = 0.7) -> Dict[str,Any]:
    last_err = None
    for attempt in range(retries + 1):
        t0 = now_ms()
        user_content, truncated = truncate_lyrics_for_budget(lyrics or "", sys_prompt, input_budget, output_budget, tokenizer)
        try:
            resp = llm.chat.completions.create(
                messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_content}],
                response_format={"type":"json_schema","json_schema":{"name":"ResoTags","schema":schema,"strict":True}},
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                request_timeout=timeout
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("empty_content")
            parsed = json.loads(content)
            dt = (now_ms() - t0) / 1000.0
            inp = estimate_tokens(sys_prompt + user_content, tokenizer)
            out = estimate_tokens(content, tokenizer)
            return {"parsed": parsed, "latency": dt, "truncated": truncated, "input_tokens_est": inp, "output_tokens_est": out}
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(last_err or "unknown_error")

def call_model_facet(llm: LLM, facet: str, canon: Dict[str,List[str]], lyrics: str, input_budget: int, output_budget: int, max_tokens: int, timeout: int, tokenizer, temperature: float) -> List[Dict[str,Any]]:
    sys_prompt = build_prompt_strict_facet(canon, facet)
    schema = build_schema_strict_facet(canon, facet)
    r = call_model(llm, sys_prompt, lyrics, input_budget, output_budget, schema, max_tokens, timeout, tokenizer, temperature, retries=1)
    arr = r["parsed"]
    if not isinstance(arr, list):
        raise RuntimeError("facet_non_array")
    out = []
    for i in arr:
        if not isinstance(i, dict):
            continue
        t = str(i.get("tag","")).strip()
        if not t:
            continue
        p = i.get("p", 0.0)
        try:
            p = float(p)
        except Exception:
            continue
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        out.append({"tag": t, "p": p})
    return out

def dedupe_topk(items: List[Dict[str,Any]], k: int) -> List[Dict[str,Any]]:
    best = {}
    for it in items:
        t = str(it.get("tag","")).strip()
        if not t:
            continue
        p = it.get("p", 0.0)
        try:
            p = float(p)
        except Exception:
            continue
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        if t not in best or p > best[t]:
            best[t] = p
    out = [{"tag": t, "p": best[t]} for t in best]
    out.sort(key=lambda x: x["p"], reverse=True)
    return out[:k]

def filter_to_canonical(parsed: Dict[str,Any], canon: Dict[str,List[str]]) -> Dict[str,List[Dict[str,Any]]]:
    out = {}
    for f in FACETS:
        items = parsed.get(f, [])
        if not isinstance(items, list):
            items = []
        cleaned = []
        for i in items:
            if not isinstance(i, dict):
                continue
            t = str(i.get("tag","")).strip()
            if t not in canon[f]:
                continue
            p = i.get("p", 0.0)
            try:
                p = float(p)
            except Exception:
                continue
            if p < 0.0:
                p = 0.0
            if p > 1.0:
                p = 1.0
            cleaned.append({"tag": t, "p": p})
        out[f] = dedupe_topk(cleaned, 5)
    return out

def ensure_minimal_from_facetwise(llm: LLM, canon: Dict[str,List[str]], lyrics: str, input_budget: int, output_budget: int, max_tokens: int, timeout: int, tokenizer, temperature: float) -> Dict[str,List[Dict[str,Any]]]:
    res = {}
    for f in FACETS:
        try:
            arr = call_model_facet(llm, f, canon, lyrics, input_budget, output_budget, max_tokens, timeout, tokenizer, temperature)
            res[f] = dedupe_topk(arr, 5)
        except Exception:
            res[f] = []
    return res

def assemble_record(tid: str, pf: Dict[str,List[Dict[str,Any]]], ok: bool, mode: str, meta: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "track_id": tid,
        "tags": {f: [o["tag"] for o in pf.get(f, [])] for f in FACETS},
        "scores": {f: {o["tag"]: o["p"] for o in pf.get(f, [])} for f in FACETS},
        "ok": bool(ok),
        "mode": mode,
        "latency": meta.get("latency"),
        "truncated": meta.get("truncated", False),
        "input_tokens_est": meta.get("input_tokens_est"),
        "output_tokens_est": meta.get("output_tokens_est"),
        "error": meta.get("error")
    }

def retag_tracks(tracks: List[Dict[str,Any]], canon: Dict[str,List[str]], model_id: str, temperature: float, input_budget: int, output_budget: int, max_tokens: int, timeout_s: int, max_workers: int, out_jsonl: Path, manifest_path: Path):
    llm = build_llm(model_id)
    tokenizer = try_load_tokenizer()
    prompt_strict = build_prompt_strict_full(canon)
    prompt_relaxed = build_prompt_relaxed_full()
    schema_strict = build_schema_strict_full(canon)
    schema_relaxed = build_schema_relaxed_full()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    in_tokens = []
    out_tokens = []
    truncated_ct = 0
    failures = 0
    failed_ids = []
    def one(t):
        tid = t.get("track_id")
        lyr = t.get("lyrics") or ""
        if not isinstance(lyr, str) or not lyr.strip():
            return assemble_record(tid, {f: [] for f in FACETS}, False, "none", {"error": "empty_lyrics"})
        try:
            r1 = call_model(llm, prompt_strict, lyr, input_budget, output_budget, schema_strict, max_tokens, timeout_s, tokenizer, temperature, retries=1)
            pf1 = filter_to_canonical(r1["parsed"], canon)
            if any(len(pf1[f]) > 0 for f in FACETS):
                return assemble_record(tid, pf1, True, "strict", r1)
            raise RuntimeError("strict_empty_after_filter")
        except Exception as e1:
            try:
                r2 = call_model(llm, prompt_strict, lyr, input_budget, output_budget, schema_strict, max_tokens, timeout_s, tokenizer, temperature, retries=1)
                pf2 = filter_to_canonical(r2["parsed"], canon)
                if any(len(pf2[f]) > 0 for f in FACETS):
                    r2["mode"] = "strict_retry"
                    return assemble_record(tid, pf2, True, "strict_retry", r2)
                raise RuntimeError("strict_retry_empty_after_filter")
            except Exception as e2:
                try:
                    r3 = call_model(llm, prompt_relaxed, lyr, input_budget, output_budget, schema_relaxed, max_tokens, timeout_s, tokenizer, temperature, retries=1)
                    pf3 = filter_to_canonical(r3["parsed"], canon)
                    if any(len(pf3[f]) > 0 for f in FACETS):
                        r3["mode"] = "relaxed"
                        return assemble_record(tid, pf3, True, "relaxed", r3)
                    raise RuntimeError("relaxed_empty_after_filter")
                except Exception as e3:
                    try:
                        pf4 = ensure_minimal_from_facetwise(llm, canon, lyr, input_budget, output_budget, max_tokens, timeout_s, tokenizer, temperature)
                        if any(len(pf4[f]) > 0 for f in FACETS):
                            return assemble_record(tid, pf4, True, "facetwise_strict", {"latency": None, "truncated": False, "input_tokens_est": None, "output_tokens_est": None})
                        return assemble_record(tid, {f: [] for f in FACETS}, False, "facetwise_strict", {"error": "facetwise_empty"})
                    except Exception as e4:
                        return assemble_record(tid, {f: [] for f in FACETS}, False, "failed", {"error": str(e4)})
    with out_jsonl.open("w", encoding="utf-8") as fo, Progress(SpinnerColumn(), TextColumn("[bold]Re-tagging[/]"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("tracks", total=len(tracks))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(one, t) for t in tracks]
            for fut in as_completed(futs):
                rec = fut.result()
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("ok") and isinstance(rec.get("latency"), (int,float)):
                    latencies.append(rec["latency"])
                if rec.get("ok") and isinstance(rec.get("input_tokens_est"), int):
                    in_tokens.append(rec["input_tokens_est"])
                if rec.get("ok") and isinstance(rec.get("output_tokens_est"), int):
                    out_tokens.append(rec["output_tokens_est"])
                if rec.get("truncated"):
                    truncated_ct += 1
                if not rec.get("ok"):
                    failures += 1
                    failed_ids.append(rec.get("track_id"))
                progress.update(task, advance=1)
    m = {
        "model": model_id,
        "variant": "CanonicalRetagWithProbs_consistent",
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "input_token_budget": input_budget,
        "output_token_budget": output_budget,
        "timeout_s": timeout_s,
        "total_input_tracks": len(tracks),
        "outputs_jsonl": str(out_jsonl),
        "truncated_contexts": truncated_ct,
        "failures": failures,
        "failed_track_ids": failed_ids,
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
    ap.add_argument("--canonical_tags", default="analysis/llm_tagging/all_tracks_unique_tags_cleaned_human_reviewed.json")
    ap.add_argument("--output_dir", default="dataset/re_tagged/deepseek-v3p1")
    ap.add_argument("--model_id", default="accounts/fireworks/models/deepseek-v3p1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--input_token_budget", type=int, default=4500)
    ap.add_argument("--output_token_budget", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env_path", type=str, default="")
    args = ap.parse_args()
    load_env(args.env_path if args.env_path else None)
    ds_path = Path(args.dataset)
    canon_path = Path(args.canonical_tags)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(ds_path)
    if args.limit and args.limit > 0:
        tracks = tracks[:args.limit]
    canon = load_canonical(canon_path)
    hdr = Table(show_header=True, header_style="bold cyan")
    hdr.add_column("Dataset")
    hdr.add_column("Total")
    hdr.add_column("Canonical Tags")
    hdr.add_row(str(ds_path), str(len(tracks)), str(canon_path))
    console.print(Panel(hdr, title="Input Summary", expand=False))
    if len(tracks) == 0:
        console.print("[yellow]No tracks to process.[/]")
        return
    out_jsonl = out_dir / (f"sample_{args.limit}.jsonl" if args.limit and args.limit > 0 else "full_run.jsonl")
    manifest = out_dir / "run_manifest.json"
    res = retag_tracks(tracks, canon, args.model_id, args.temperature, args.input_token_budget, args.output_token_budget, args.max_tokens, args.timeout_s, args.max_workers, out_jsonl, manifest)
    summ = Table(show_header=False, box=None)
    summ.add_row("Outputs", str(out_jsonl))
    summ.add_row("Manifest", str(manifest))
    summ.add_row("Tracks Processed", str(len(tracks)))
    summ.add_row("Avg Latency (s)", "-" if res["latency_avg_s"] is None else str(res["latency_avg_s"]))
    summ.add_row("Avg Input Tokens", "-" if res["input_tokens_avg"] is None else str(res["input_tokens_avg"]))
    summ.add_row("Avg Output Tokens", "-" if res["output_tokens_avg"] is None else str(res["output_tokens_avg"]))
    summ.add_row("Failures", str(res.get("failures", 0)))
    console.print(Panel(summ, title="Run Summary", expand=False))

if __name__ == "__main__":
    main()
