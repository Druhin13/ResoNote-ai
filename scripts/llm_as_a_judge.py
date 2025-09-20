#!/usr/bin/env python3
"""
llm-as-a-judge.py
"""

import os
import re
import json
import time
import math
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from fireworks import LLM
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

# -------------------- setup --------------------

console = Console()
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise RuntimeError("FIREWORKS_API_KEY not set.")

FACET_KEYS = ["Emotional_Tone", "Thematic_Content", "Narrative_Structure", "Lyrical_Style"]

CANONICAL_OVERRIDES: Dict[str, str] = {
    "llama-v3p1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "fireworks/llama-v3p1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
}

def safe_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

def canonicalize_model_id(model: str) -> str:
    key = model.strip()
    if key in CANONICAL_OVERRIDES:
        canon = CANONICAL_OVERRIDES[key]
        if canon != key:
            console.print(f"Using canonical id for {key}: {canon}")
        return canon
    return key

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

def now_ms():
    return int(time.time() * 1000)

# -------------------- data IO --------------------

def load_tracks(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_all_tags_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, List[str]]]:
    by_id: Dict[str, Dict[str, List[str]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("track_id")
            tags = obj.get("tags") or {}
            if isinstance(tid, str) and isinstance(tags, dict):
                by_id[tid] = {k: list(v) for k, v in tags.items() if k in FACET_KEYS and isinstance(v, list)}
    return by_id

def load_canonical(path: Path) -> Dict[str, set]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k in FACET_KEYS:
        out[k] = set([str(x).strip() for x in obj.get(k, []) if isinstance(x, str) and x.strip()])
    return out

def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# -------------------- bucketing (for sanity prints) --------------------

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

def combo_key(t):
    return f"{bin_valence(t.get('valence'))}|{bin_popularity(t.get('popularity'))}|{bin_energy(t.get('energy'))}"

# -------------------- TSV protocol --------------------

def make_tsv_spec(reason_chars: int) -> str:
    return (
        "Return one line per CANDIDATE TAG, using TAB-separated fields exactly:\n"
        "<facet>\t<tag>\t<agree: Y|N>\t<confidence: 0..1>\t<reason ≤"
        + str(reason_chars) +
        " chars>\t[optional <suggest_if_disagree>]\n"
        "Rules:\n"
        " • facet must be one of: Emotional_Tone, Thematic_Content, Narrative_Structure, Lyrical_Style\n"
        " • tag must be copied EXACTLY from Candidates and you must keep the same ORDER as in Candidates\n"
        " • agree is 'Y' if the lyric supports the tag; 'N' otherwise\n"
        " • confidence is a number in [0,1]\n"
        " • reason must be concise (≤" + str(reason_chars) + " chars) and contain no tabs\n"
        " • If you disagree and know a better replacement from our canonical set, add it as a 6th field; otherwise omit\n"
        "Output ONLY these lines. No preface, no code fences, no extra JSON."
    )

def parse_tsv_lines(text: str) -> List[List[str]]:
    lines = []
    if not text:
        return lines
    m = re.search(r"```(?:tsv|text)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1)
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "\t" in s:
            parts = s.split("\t")
        elif " | " in s:
            parts = s.split(" | ")
        else:
            parts = re.split(r"\s{2,}", s)
        parts = [p.strip() for p in parts if p is not None]
        if len(parts) >= 5:
            lines.append(parts[:6])
    return lines

def coerce_bool_from_yn(x: str) -> bool:
    return str(x).strip().upper() in ("Y", "YES", "TRUE", "T", "1")

def coerce_float_01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))

def truncate_reason(s: Any, maxlen: int) -> str:
    if not isinstance(s, str):
        return "unspecified"
    s = s.strip()
    if len(s) <= maxlen:
        return s
    return s[:maxlen]

def format_candidate_block(cand: Dict[str, List[str]]) -> str:
    return json.dumps({k: cand.get(k, []) for k in FACET_KEYS}, ensure_ascii=False, separators=(",", ":"))

def build_pairs(candidates: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    pairs = []
    for facet in FACET_KEYS:
        for tag in candidates.get(facet, []):
            pairs.append((facet, tag))
    return pairs

def normalize_from_tsv(tsv_text: str, candidates: Dict[str, List[str]], canonical: Dict[str, set], reason_chars: int) -> Tuple[Dict[str, Any], float]:
    expected_pairs = build_pairs(candidates)
    idx = {(facet, tag): i for i, (facet, tag) in enumerate(expected_pairs)}
    placed = [False] * len(expected_pairs)
    records = {k: [{"tag": tag, "agree": False, "confidence": 0.0, "reason": "not returned"} for tag in candidates.get(k, [])] for k in FACET_KEYS}

    lines = parse_tsv_lines(tsv_text)
    parsed = 0
    for parts in lines:
        facet, tag = parts[0] if len(parts)>0 else "", parts[1] if len(parts)>1 else ""
        agree_raw = parts[2] if len(parts)>2 else "N"
        conf_raw = parts[3] if len(parts)>3 else "0"
        reason_raw = parts[4] if len(parts)>4 else "unspecified"
        sugg_raw = parts[5] if len(parts)>5 else None

        facet = str(facet).strip()
        tag = str(tag).strip()
        if facet not in FACET_KEYS:
            continue
        if tag not in candidates.get(facet, []):
            continue

        agree = coerce_bool_from_yn(agree_raw)
        conf = coerce_float_01(conf_raw)
        reason = truncate_reason(reason_raw, maxlen=reason_chars)

        rec = {"tag": tag, "agree": agree, "confidence": conf, "reason": reason}
        if (not agree) and isinstance(sugg_raw, str) and sugg_raw.strip() and (sugg_raw.strip() in canonical.get(facet, set())):
            rec["suggest_if_disagree"] = sugg_raw.strip()

        pos = candidates[facet].index(tag)
        records[facet][pos] = rec
        if (facet, tag) in idx and not placed[idx[(facet, tag)]]:
            placed[idx[(facet, tag)]] = True
            parsed += 1

    parsed_fraction = (parsed / max(1, len(expected_pairs)))
    out = {"decisions": records, "meta": {"version": "p2.tsv.v2"}}
    return out, parsed_fraction

# -------------------- prompts --------------------

def prompt_variants(tsv_spec: str) -> Dict[str, str]:
    # Evidence-first, stable framing; slight agree-lean only when plausible
    core = (
        "You are an evidence-based lyric tag judge. Decide using only the provided lyrics. "
        "Agree (Y) when there is explicit or clearly plausible support; disagree (N) when contradicted or unsupported. "
        "Use confidence to reflect strength of evidence. Be concise."
    )
    framing = (
        "For each candidate tag under each facet, output exactly one line in the order given, "
        "copying the tag text verbatim. " + tsv_spec
    )
    v0 = core + " If in doubt, keep confidence low." + " " + framing
    v1 = core + " Prefer agree with low confidence when support is plausible, not vibe-only." + " " + framing
    v2 = core + " Confidence guide: explicit=0.8–1.0; plausible=0.55–0.70; weak/none=0–0.4." + " " + framing
    return {"V0_AgreeLean": v0, "V1_Compact": v1, "V2_ConfidenceGuided": v2}

# -------------------- LLM plumbing --------------------

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

def build_serverless_llm(model_id: str) -> LLM:
    canon = canonicalize_model_id(model_id)
    ok, detail, stream_only = ping_serverless_model(canon)
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Model"); t.add_column("Status"); t.add_column("Details")
    if ok and not stream_only:
        t.add_row(model_id, "[green]ready[/]", "-")
        console.print(Panel(t, title="Serverless Model Availability", expand=False))
        return LLM(model=canon, deployment_type="serverless", api_key=FIREWORKS_API_KEY)
    elif ok and stream_only:
        t.add_row(model_id, "[yellow]ready (stream-only)[/]", detail or "-")
    else:
        t.add_row(model_id, "[red]unavailable[/]", detail or "-")
    console.print(Panel(t, title="Serverless Model Availability", expand=False))
    raise RuntimeError(f"Model not available: {model_id} ({detail})")

def call_judge(
    llm: LLM,
    sys_prompt: str,
    lyrics: str,
    candidates: Dict[str, List[str]],
    input_budget: int,
    output_budget: int,
    max_tokens: int,
    timeout: int,
    tokenizer: Any,
    temperature: float,
    canonical: Dict[str, set],
    reason_chars: int,
    tsv_spec: str,
    retries: int = 2
):
    last_err = None
    last_raw = None
    for attempt in range(retries + 1):
        t0 = now_ms()
        candidate_json = format_candidate_block(candidates)
        # budgeted truncation
        header = "CANDIDATES:\n" + candidate_json + "\n\nLYRICS:\n"
        chunks = (lyrics or "").splitlines()
        lo, hi = 0, len(chunks)
        best = ""
        total_budget = max(0, input_budget - output_budget)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = "\n".join(chunks[:mid])
            content = header + candidate + "\n\n" + tsv_spec
            toks = estimate_tokens(sys_prompt + content, tokenizer)
            if toks <= total_budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        user_content = "CANDIDATES:\n" + candidate_json + "\n\nLYRICS:\n" + best + "\n\n" + tsv_spec
        truncated = (best != "\n".join(chunks))

        try:
            resp = llm.chat.completions.create(
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user_content}],
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                request_timeout=timeout
            )
            content = resp.choices[0].message.content
            last_raw = content
            dt = (now_ms() - t0) / 1000.0
            if not content:
                raise ValueError("empty content")

            normalized, parsed_frac = normalize_from_tsv(content, candidates, canonical, reason_chars=reason_chars)
            inp_toks = estimate_tokens(sys_prompt + user_content, tokenizer)
            out_toks = estimate_tokens(content, tokenizer)
            hit_cap = (out_toks >= 0.95 * max_tokens)

            ok_flag = parsed_frac >= 0.8
            return {
                "ok": ok_flag,
                "parsed": normalized,
                "raw": content,
                "latency": dt,
                "truncated": truncated,
                "input_tokens_est": inp_toks,
                "output_tokens_est": out_toks,
                "hit_output_cap": hit_cap,
                "parsed_fraction": parsed_frac,
                "error": None if ok_flag else f"parsed_frac={parsed_frac:.2f}"
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(0.6 * (attempt + 1))

    normalized, parsed_frac = normalize_from_tsv("", candidates, canonical, reason_chars=reason_chars)
    return {"ok": False, "parsed": normalized, "raw": last_raw, "latency": None,
            "truncated": False, "input_tokens_est": None, "output_tokens_est": None,
            "hit_output_cap": False, "parsed_fraction": parsed_frac,
            "error": last_err or "unknown_error"}

# -------------------- metrics --------------------

def eval_outputs(dev_results: List[Dict[str, Any]], candidates_map: Dict[str, Dict[str, List[str]]]):
    total = len(dev_results)
    ok = 0
    latencies = []
    inp = []
    out = []
    trunc = 0
    cap_hits = 0
    parsed_fracs = []

    agree_counts = []
    disagree_counts = []
    agree_conf = []
    disagree_conf = []

    coverage_ok = 0

    # autofill counters
    autofill_total = 0
    decision_total = 0
    autofill_by_facet = {k: 0 for k in FACET_KEYS}
    decision_by_facet = {k: 0 for k in FACET_KEYS}

    for r in dev_results:
        parsed = r.get("parsed") or {}
        dec = parsed.get("decisions") or {}
        if r.get("ok"):
            ok += 1
        if isinstance(r.get("latency"), (int, float)):
            latencies.append(r["latency"])
        if isinstance(r.get("input_tokens_est"), int):
            inp.append(r["input_tokens_est"])
        if isinstance(r.get("output_tokens_est"), int):
            out.append(r["output_tokens_est"])
        if r.get("truncated"):
            trunc += 1
        if r.get("hit_output_cap"):
            cap_hits += 1
        if isinstance(r.get("parsed_fraction"), (int, float)):
            parsed_fracs.append(float(r["parsed_fraction"]))

        tid = r.get("track_id")
        candidates = candidates_map.get(tid, {})
        full_cov = True
        a_cnt = 0
        d_cnt = 0
        for k in FACET_KEYS:
            cand_list = candidates.get(k, [])
            decided = dec.get(k, [])
            tag_set = set([c.get("tag") for c in decided if isinstance(c, dict) and isinstance(c.get("tag"), str)])
            want = set(cand_list)
            if tag_set != want:
                full_cov = False
            for item in decided:
                if not isinstance(item, dict):
                    continue
                decision_by_facet[k] += 1
                decision_total += 1
                if item.get("reason") == "not returned":
                    autofill_by_facet[k] += 1
                    autofill_total += 1
                cf = item.get("confidence")
                if item.get("agree") is True:
                    a_cnt += 1
                    if isinstance(cf, (int, float)):
                        agree_conf.append(float(cf))
                elif item.get("agree") is False:
                    d_cnt += 1
                    if isinstance(cf, (int, float)):
                        disagree_conf.append(float(cf))
        if full_cov:
            coverage_ok += 1
        agree_counts.append(a_cnt)
        disagree_counts.append(d_cnt)

    json_ok_rate = ok / total if total > 0 else 0.0
    coverage_rate = coverage_ok / total if total > 0 else 0.0
    p50 = None
    p95 = None
    if latencies:
        p50 = sorted(latencies)[len(latencies)//2]
        p95 = sorted(latencies)[max(0, math.ceil(0.95*len(latencies))-1)]
    avg_inp = sum(inp)/len(inp) if inp else None
    avg_out = sum(out)/len(out) if out else None
    avg_agree = sum(agree_counts)/len(agree_counts) if agree_counts else 0.0
    avg_disagree = sum(disagree_counts)/len(disagree_counts) if disagree_counts else 0.0
    total_decisions = avg_agree + avg_disagree if (avg_agree + avg_disagree) > 0 else 1.0
    agree_rate = avg_agree/total_decisions
    d_agree = sum(agree_conf)/len(agree_conf) if agree_conf else None
    d_disagree = sum(disagree_conf)/len(disagree_conf) if disagree_conf else None

    cap_hit_fraction = cap_hits / total if total > 0 else 0.0
    avg_parsed_fraction = sum(parsed_fracs)/len(parsed_fracs) if parsed_fracs else 0.0
    autofill_rate_overall = (autofill_total / decision_total) if decision_total > 0 else 0.0
    autofill_rate_by_facet = {k: (autofill_by_facet[k]/decision_by_facet[k] if decision_by_facet[k] > 0 else 0.0) for k in FACET_KEYS}

    return {
        "json_ok_rate": round(json_ok_rate, 4),
        "coverage_rate": round(coverage_rate, 4),
        "latency_p50_s": None if p50 is None else round(p50, 3),
        "latency_p95_s": None if p95 is None else round(p95, 3),
        "input_tokens_avg": None if avg_inp is None else round(avg_inp, 1),
        "output_tokens_avg": None if avg_out is None else round(avg_out, 1),
        "avg_agree_per_track": round(avg_agree, 3),
        "avg_disagree_per_track": round(avg_disagree, 3),
        "agree_rate": round(agree_rate, 4),
        "agree_conf_avg": None if d_agree is None else round(d_agree, 3),
        "disagree_conf_avg": None if d_disagree is None else round(d_disagree, 3),
        "truncated_fraction": round((trunc/total) if total>0 else 0.0, 4),
        "cap_hit_fraction": round(cap_hit_fraction, 4),
        "avg_parsed_fraction": round(avg_parsed_fraction, 4),
        "autofill_rate_overall": round(autofill_rate_overall, 4),
        "autofill_rate_by_facet": {k: round(v, 4) for k, v in autofill_rate_by_facet.items()},
        "total_tracks": total,
        "ok_tracks": ok
    }

# -------------------- orchestration --------------------

def run_eval(
    eval_tracks: List[Dict[str, Any]],
    candidates_map: Dict[str, Dict[str, List[str]]],
    llm: LLM,
    sys_prompt: str,
    input_budget: int,
    output_budget: int,
    max_workers: int,
    timeout: int,
    max_tokens: int,
    tokenizer: Any,
    temperature: float,
    canonical: Dict[str, set],
    reason_chars: int,
    tsv_spec: str,
    resume_from: Path = None
):
    # Resume support: load previous results and skip processed ids
    previous: Dict[str, Dict[str, Any]] = {}
    if resume_from and resume_from.exists():
        try:
            prev_list = json.loads(resume_from.read_text(encoding="utf-8"))
            for item in prev_list:
                if isinstance(item, dict) and "track_id" in item:
                    previous[item["track_id"]] = item
            console.print(Panel(f"Resuming: found {len(previous)} prior results in {resume_from}", title="Resume", expand=False))
        except Exception as e:
            console.print(Panel(f"Could not read resume file: {e}", title="Resume", expand=False))

    todo = []
    for t in eval_tracks:
        tid = t.get("track_id")
        if tid in previous:
            continue
        if tid in candidates_map:
            todo.append(t)
    console.print(Panel(f"To process: {len(todo)} (skipping {len(previous)} already done)", title="Worklist", expand=False))

    results = list(previous.values())

    def one(tobj):
        tid = tobj.get("track_id")
        lyrics = tobj.get("lyrics") or ""
        cands = candidates_map.get(tid, {k: [] for k in FACET_KEYS})
        r = call_judge(llm, sys_prompt, lyrics, cands, input_budget, output_budget,
                       max_tokens, timeout, tokenizer, temperature, canonical,
                       reason_chars=reason_chars, tsv_spec=tsv_spec)
        return {
            "track_id": tid,
            "ok": r["ok"],
            "error": r.get("error"),
            "parsed": r.get("parsed"),
            "raw": r.get("raw"),
            "latency": r.get("latency"),
            "truncated": r.get("truncated"),
            "input_tokens_est": r.get("input_tokens_est"),
            "output_tokens_est": r.get("output_tokens_est"),
            "hit_output_cap": r.get("hit_output_cap"),
            "parsed_fraction": r.get("parsed_fraction"),
        }

    with Progress(SpinnerColumn(), TextColumn("[bold]LLM-as-a-Judge[/]: {task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("tracks", total=len(todo))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(one, t) for t in todo]
            for f in as_completed(futs):
                results.append(f.result())
                progress.update(task, advance=1)

    # Sort results by track_id for stability
    results_sorted = sorted(results, key=lambda x: str(x.get("track_id", "")))

    metrics = eval_outputs(results_sorted, {t.get("track_id"): candidates_map.get(t.get("track_id"), {}) for t in eval_tracks if t.get("track_id") in candidates_map})
    return results_sorted, metrics

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    # inputs
    ap.add_argument("--eval_path", type=str, default="analysis/llm_selection_p1/splits/eval.json")
    ap.add_argument("--tags_jsonl", type=str, default="dataset/re_tagged/deepseek-v3p1/all_tracks.jsonl")
    ap.add_argument("--canonical_path", type=str, default="analysis/llm_tagging/all_tracks_unique_tags_cleaned_human_reviewed.json")
    # frozen config (defaults chosen for research-grade stability)
    ap.add_argument("--model_id", type=str, default="llama-v3p1-70b-instruct")
    ap.add_argument("--variant_name", type=str, default="V0_AgreeLean", choices=["V0_AgreeLean","V1_Compact","V2_ConfidenceGuided"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reason_chars", type=int, default=140)
    ap.add_argument("--max_tokens", type=int, default=1024)
    # runtime / budgets
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--input_token_budget", type=int, default=3500)
    ap.add_argument("--output_token_budget", type=int, default=512)
    ap.add_argument("--timeout_s", type=int, default=120)
    # sampling & resume
    ap.add_argument("--sample_n", type=int, default=0, help="If >0, run on a random subset for a smoke test")
    ap.add_argument("--sample_seed", type=int, default=1337)
    ap.add_argument("--resume", type=str, default="", help="Optional path to an existing eval_outputs.json to resume from")
    # outputs
    ap.add_argument("--outdir", type=str, default="analysis/llm_as_judge")
    args = ap.parse_args()

    tokenizer = try_load_tokenizer()

    # Load inputs
    eval_tracks = load_tracks(Path(args.eval_path))
    all_tags_map = load_all_tags_jsonl(Path(args.tags_jsonl))
    canonical = load_canonical(Path(args.canonical_path))

    present = [t for t in eval_tracks if t.get("track_id") in all_tags_map]
    if args.sample_n and args.sample_n > 0:
        random.Random(args.sample_seed).shuffle(present)
        present = present[:args.sample_n]

    missing = [t for t in eval_tracks if t.get("track_id") not in all_tags_map]
    hdr = Table(show_header=True, header_style="bold blue")
    hdr.add_column("Eval Tracks"); hdr.add_column("With Candidates (used)"); hdr.add_column("Missing Candidates")
    hdr.add_row(str(len(eval_tracks)), str(len(present)), str(len(missing)))
    console.print(Panel(hdr, title="Input Coverage", expand=False))
    if missing:
        miss_ids = [t.get("track_id") for t in missing[:20]]
        console.print(Panel(("Missing track_ids (sample): " + ", ".join(miss_ids) + (" ..." if len(missing) > 20 else "")), title="Warning", expand=False))
    if present:
        bucket_counts = Counter(combo_key(t) for t in present)
        console.print(Panel(str(bucket_counts.most_common(10)), title="Stratification Buckets (used)", expand=False))

    # Build model and prompt
    llm = build_serverless_llm(args.model_id)
    tsv_spec = make_tsv_spec(args.reason_chars)
    variants = prompt_variants(tsv_spec)
    sys_prompt = variants[args.variant_name]

    # Out paths
    outdir = Path(args.outdir) / safe_name(args.model_id) / safe_name(args.variant_name) / f"T_{str(args.temperature).replace('.','_')}"
    outdir.mkdir(parents=True, exist_ok=True)
    outputs_path = outdir / "eval_outputs.json"
    metrics_path = outdir / "eval_metrics.json"

    # Run
    results, metrics = run_eval(
        present,
        all_tags_map,
        llm,
        sys_prompt,
        args.input_token_budget,
        args.output_token_budget,
        args.max_workers,
        args.timeout_s,
        args.max_tokens,
        tokenizer,
        args.temperature,
        canonical,
        args.reason_chars,
        tsv_spec,
        resume_from=Path(args.resume) if args.resume else None
    )

    # Save
    save_json(outputs_path, results)
    payload = {
        "config": {
            "model_id": args.model_id,
            "variant_name": args.variant_name,
            "temperature": args.temperature,
            "reason_chars": args.reason_chars,
            "max_tokens": args.max_tokens,
            "input_token_budget": args.input_token_budget,
            "output_token_budget": args.output_token_budget,
            "timeout_s": args.timeout_s,
        },
        "paths": {
            "eval_path": str(Path(args.eval_path)),
            "tags_jsonl": str(Path(args.tags_jsonl)),
            "canonical_path": str(Path(args.canonical_path)),
            "outputs_path": str(outputs_path)
        },
        "metrics": metrics
    }
    save_json(metrics_path, payload)

    # Print summary
    t = Table(show_header=True, header_style="bold green")
    t.add_column("JSON OK"); t.add_column("Coverage"); t.add_column("AgreeRate")
    t.add_column("Autofill%"); t.add_column("CapHit%"); t.add_column("p95(s)")
    t.add_row(
        f"{metrics['json_ok_rate']:.2f}",
        f"{metrics['coverage_rate']:.2f}",
        f"{metrics['agree_rate']:.2f}",
        f"{metrics['autofill_rate_overall']*100:.1f}",
        f"{metrics['cap_hit_fraction']*100:.1f}",
        "-" if metrics["latency_p95_s"] is None else f"{metrics['latency_p95_s']:.3f}",
    )
    console.print(Panel(t, title="LLM-as-a-Judge — Eval Summary", expand=False))

    final = Table(show_header=False, box=None)
    final.add_row("Eval Input", str(Path(args.eval_path)))
    final.add_row("Candidates JSONL", str(Path(args.tags_jsonl)))
    final.add_row("Canonical Tags", str(Path(args.canonical_path)))
    final.add_row("Output Dir", str(outdir))
    final.add_row("Outputs", str(outputs_path))
    final.add_row("Metrics", str(metrics_path))
    console.print(Panel(final, title="Done", expand=False))

if __name__ == "__main__":
    main()
