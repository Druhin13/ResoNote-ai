#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple
from functools import lru_cache

WS = re.compile(r"\s+")
UNDERSCORES = re.compile(r"_+")
DASHES = re.compile(r"-{2,}")
DESCR_SUFFIX = re.compile(r"^(?P<base>.+?)-(?:like|driven|focused|based|oriented|centric|led)$")
ADJISH_SUFFIX = re.compile(r".*(?:ic|ive|al|ous|ful|less|ish|ary|ory|ate|able|ible|y)$")

def normalize_basic(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = WS.sub("-", s)
    s = UNDERSCORES.sub("-", s)
    s = DASHES.sub("-", s)
    s = s.strip("-")
    return s

def _ensure_nltk_pos_and_wn():
    import nltk
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    from nltk import pos_tag
    from nltk.corpus import wordnet as wn
    return pos_tag, wn

@lru_cache(maxsize=100_000)
def _pos_of(tag: str, pos_tag) -> str:
    try:
        return pos_tag([tag])[0][1]
    except Exception:
        return ""

@lru_cache(maxsize=100_000)
def _is_wordnet_noun(word: str, wn) -> bool:
    try:
        return bool(wn.synsets(word, pos="n"))
    except Exception:
        return False

@lru_cache(maxsize=100_000)
def _is_wordnet_adj(word: str, wn) -> bool:
    try:
        return bool(wn.synsets(word, pos="a") or wn.synsets(word, pos="s"))
    except Exception:
        return False

def singularize_whole_tag(tag: str, p_inflect, pos_tag, wn) -> str:
    if not tag or len(tag) < 3:
        return tag
    pos = _pos_of(tag, pos_tag)
    if pos not in ("NNS", "NNPS"):
        return tag
    cand = p_inflect.singular_noun(tag)
    if not cand or cand == tag:
        return tag
    if not _is_wordnet_noun(cand, wn):
        return tag
    if p_inflect.plural(cand) != tag:
        return tag
    return cand

def singularize_hyphen_last_segment(tag: str, p_inflect, wn) -> str:
    if "-" not in tag:
        return tag
    parts = tag.split("-")
    last = parts[-1]
    if len(last) < 2:
        return tag
    cand = p_inflect.singular_noun(last)
    if not cand or cand == last:
        return tag
    if not _is_wordnet_noun(cand, wn):
        return tag
    if p_inflect.plural(cand) != last:
        return tag
    parts[-1] = cand
    return "-".join(parts)

def _adjective_candidates_from_wordnet(tag: str, wn) -> List[str]:
    surfaces = {tag}
    if "-" in tag:
        surfaces.add(tag.replace("-", "_"))
    adjs = set()
    for surface in surfaces:
        try:
            syns = wn.synsets(surface)
        except Exception:
            syns = []
        for ss in syns:
            for lem in ss.lemmas():
                for rel in lem.derivationally_related_forms():
                    pos = rel.synset().pos()
                    if pos in ("a", "s"):
                        name = rel.name().lower().replace("_", "-")
                        adjs.add(name)
    return sorted(adjs)

def _noun_candidates_from_wordnet(tag: str, wn) -> List[str]:
    surfaces = {tag}
    if "-" in tag:
        surfaces.add(tag.replace("-", "_"))
    nouns = set()
    for surface in surfaces:
        try:
            syns = wn.synsets(surface)
        except Exception:
            syns = []
        for ss in syns:
            for lem in ss.lemmas():
                for rel in lem.derivationally_related_forms():
                    if rel.synset().pos() == "n":
                        nouns.add(rel.name().lower().replace("_", "-"))
    return sorted(nouns)

def _score_adj(adj: str, wn) -> int:
    try:
        return len(wn.synsets(adj, pos="a")) + len(wn.synsets(adj, pos="s"))
    except Exception:
        return 0

def _score_noun(noun: str, wn) -> int:
    try:
        return len(wn.synsets(noun, pos="n"))
    except Exception:
        return 0

def _dedup_with_hyphen_preference(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    key_map: Dict[str, Tuple[str, int]] = {}
    for t in seq:
        k = t.replace("-", "")
        if k not in key_map:
            idx = len(out)
            key_map[k] = (t, idx)
            out.append(t)
        else:
            cur, idx = key_map[k]
            if "-" in t and "-" not in cur:
                out[idx] = t
                key_map[k] = (t, idx)
    return out

def _is_hyphen_adjective(tag: str, wn) -> bool:
    if _is_wordnet_adj(tag, wn):
        return True
    if "-" in tag:
        last = tag.split("-")[-1]
        if _is_wordnet_adj(last, wn):
            return True
    return False

def _is_hyphen_noun(tag: str, wn) -> bool:
    if _is_wordnet_noun(tag, wn):
        return True
    if "-" in tag:
        last = tag.split("-")[-1]
        if _is_wordnet_noun(last, wn):
            return True
    return False

def convert_emotional_tone_to_adjectives(tags: Iterable[str], wn) -> List[str]:
    out: List[str] = []
    seen_keys: Dict[str, Tuple[str, int]] = {}
    for t in tags:
        cands = _adjective_candidates_from_wordnet(t, wn)
        best: Optional[str] = None
        if cands:
            scored = sorted(((c, _score_adj(c, wn), len(c)) for c in cands), key=lambda x: (-x[1], x[2], x[0]))
            if scored and scored[0][1] > 0:
                best = scored[0][0]
        replacement = best if best else t
        k = replacement.replace("-", "")
        if k not in seen_keys:
            idx = len(out)
            seen_keys[k] = (replacement, idx)
            out.append(replacement)
        else:
            cur, idx = seen_keys[k]
            if "-" in replacement and "-" not in cur:
                out[idx] = replacement
                seen_keys[k] = (replacement, idx)
    out = [t for t in out if _is_hyphen_adjective(t, wn)]
    out = _dedup_with_hyphen_preference(out)
    return out

@lru_cache(maxsize=100_000)
def _derivational_adjectives(word: str, wn) -> List[str]:
    adjs = set()
    surfaces = {word}
    if "-" in word:
        surfaces.add(word.replace("-", "_"))
    for surface in surfaces:
        try:
            syns = wn.synsets(surface)
        except Exception:
            syns = []
        for ss in syns:
            for lem in ss.lemmas():
                for rel in lem.derivationally_related_forms():
                    if rel.synset().pos() in ("a", "s"):
                        adjs.add(rel.name().lower().replace("_", "-"))
    return sorted(adjs)

def drop_nouns_with_adj_siblings_keep_adjs_order(tags: List[str], wn) -> List[str]:
    present = set(tags)
    out: List[str] = []
    for t in tags:
        if _is_hyphen_adjective(t, wn):
            out.append(t)
            continue
        rel_adjs = _derivational_adjectives(t, wn)
        if any(a in present for a in rel_adjs):
            continue
        if _is_hyphen_adjective(t, wn):
            out.append(t)
    return out

def collapse_ing_when_non_ing_sibling_exists(tags: List[str], wn) -> List[str]:
    present = set(tags)
    keep = []
    for t in tags:
        if t.endswith("ing"):
            rel_adjs = set(_derivational_adjectives(t, wn))
            non_ing_siblings = [a for a in rel_adjs if a in present and not a.endswith("ing")]
            if non_ing_siblings:
                continue
        keep.append(t)
    return keep

def reduce_descriptor_compound(tag: str) -> str:
    prev = tag
    while True:
        m = DESCR_SUFFIX.match(prev)
        if not m:
            return prev
        prev = m.group("base")

def generate_morph_noun_candidates(word: str) -> List[str]:
    c = set()
    if word.endswith("ative"):
        c.add(word[:-5] + "ation")
    if word.endswith("tive"):
        c.add(word[:-4] + "tion")
    if word.endswith("sive"):
        c.add(word[:-4] + "sion")
    if word.endswith("ive"):
        c.add(word[:-3] + "ion")
        c.add(word + "ness")
    if word.endswith("al"):
        c.add(word[:-2] + "ality")
        c.add(word[:-2] + "ation")
    if word.endswith("ic"):
        if len(word) > 2:
            c.add(word[:-2])
            c.add(word[:-2] + "ics")
    if word.endswith("ous"):
        c.add(word[:-3] + "ity")
        c.add(word[:-3] + "ousness")
    if word.endswith("ful"):
        c.add(word[:-3] + "fulness")
    if word.endswith("less"):
        c.add(word[:-4] + "lessness")
    if word.endswith("y") and len(word) > 3:
        c.add(word[:-1] + "iness")
    if word.endswith("ing") and len(word) > 4:
        c.add(word[:-3])
        c.add(word[:-3] + "ingness")
    return sorted(c)

def morphological_fallback_to_noun(tag: str, wn) -> Optional[str]:
    if "-" in tag:
        base = tag
    else:
        base = tag
    candidates = generate_morph_noun_candidates(base)
    scored = []
    for cand in candidates:
        if _is_wordnet_noun(cand, wn):
            scored.append((cand, _score_noun(cand, wn), len(cand)))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[1], x[2], x[0]))
    return scored[0][0]

def descriptor_reduce_list(tags: Iterable[str]) -> List[str]:
    out = []
    seen: Dict[str, Tuple[str, int]] = {}
    for t in tags:
        r = reduce_descriptor_compound(t)
        k = r.replace("-", "")
        if k not in seen:
            idx = len(out)
            seen[k] = (r, idx)
            out.append(r)
        else:
            cur, idx = seen[k]
            if "-" in r and "-" not in cur:
                out[idx] = r
                seen[k] = (r, idx)
    return out

def filter_non_noun_descriptor_shaped(tags: Iterable[str], wn) -> List[str]:
    out = []
    for t in tags:
        if _is_hyphen_noun(t, wn):
            out.append(t)
            continue
        if DESCR_SUFFIX.match(t) or ADJISH_SUFFIX.match(t):
            continue
        out.append(t)
    return out

def convert_to_nouns_enforce(tags: Iterable[str], wn) -> List[str]:
    out: List[str] = []
    seen: Dict[str, Tuple[str, int]] = {}
    for t in tags:
        rep = t
        if not _is_hyphen_noun(rep, wn):
            noun_cands = _noun_candidates_from_wordnet(rep, wn)
            best = None
            if noun_cands:
                scored = sorted(((c, _score_noun(c, wn), len(c)) for c in noun_cands), key=lambda x: (-x[1], x[2], x[0]))
                if scored and scored[0][1] > 0:
                    best = scored[0][0]
            if not best:
                mf = morphological_fallback_to_noun(rep, wn)
                if mf:
                    best = mf
            if best:
                rep = best
        k = rep.replace("-", "")
        if k not in seen:
            idx = len(out)
            seen[k] = (rep, idx)
            out.append(rep)
        else:
            cur, idx = seen[k]
            if "-" in rep and "-" not in cur:
                out[idx] = rep
                seen[k] = (rep, idx)
    out = _dedup_with_hyphen_preference(out)
    return out

def load_json_obj(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Source JSON must be an object mapping facets to lists.")
    for k, v in list(data.items()):
        if not isinstance(v, list):
            data[k] = []
    return data

def save_json_obj(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def process(src: Path, dst: Path) -> Path:
    try:
        import inflect
    except Exception:
        raise RuntimeError("This script requires `inflect`. Install with: pip install inflect")
    pos_tag, wn = _ensure_nltk_pos_and_wn()
    p = inflect.engine()
    data = load_json_obj(src)
    cleaned: Dict[str, List[str]] = {}
    for facet, vals in data.items():
        key_map: Dict[str, Tuple[str, int]] = {}
        out: List[str] = []
        for raw in vals:
            if not isinstance(raw, str):
                continue
            tag = normalize_basic(raw)
            if not tag:
                continue
            tag2 = singularize_whole_tag(tag, p, pos_tag, wn)
            if tag2 == tag:
                tag2 = singularize_hyphen_last_segment(tag2, p, wn)
            k = tag2.replace("-", "")
            if k not in key_map:
                idx = len(out)
                key_map[k] = (tag2, idx)
                out.append(tag2)
            else:
                cur, idx = key_map[k]
                if "-" in tag2 and "-" not in cur:
                    out[idx] = tag2
                    key_map[k] = (tag2, idx)
        cleaned[facet] = out
    if "Emotional_Tone" in cleaned:
        et = convert_emotional_tone_to_adjectives(cleaned["Emotional_Tone"], wn)
        et = drop_nouns_with_adj_siblings_keep_adjs_order(et, wn)
        et = collapse_ing_when_non_ing_sibling_exists(et, wn)
        et = _dedup_with_hyphen_preference(et)
        cleaned["Emotional_Tone"] = et
    for facet in ("Thematic_Content", "Narrative_Structure", "Lyrical_Style"):
        if facet in cleaned:
            step = descriptor_reduce_list(cleaned[facet])
            step = convert_to_nouns_enforce(step, wn)
            step = descriptor_reduce_list(step)
            step = filter_non_noun_descriptor_shaped(step, wn)
            step = _dedup_with_hyphen_preference(step)
            cleaned[facet] = step
    save_json_obj(dst, cleaned)
    return dst

def main():
    ap = argparse.ArgumentParser(description="Facet-aware tag cleaner with descriptor reduction and noun/adjective enforcement.")
    ap.add_argument("--src", type=Path, required=False, default="analysis/llm_selection_p1/final_tags/aggregates/all_unique_tags.json")
    ap.add_argument("--dst", type=Path, required=False, default="analysis/llm_selection_p1/final_tags/aggregates/cleaned/all_unique_tags_cleaned.json")
    args = ap.parse_args()
    out = process(args.src, args.dst)
    print(str(out))

if __name__ == "__main__":
    main()
