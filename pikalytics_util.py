"""
Utilities for reading Pikalytics usage data (HTML) for team building.

Works offline-first via cache files in pikalytics_cache/.

Files expected (you can Save As from your browser into that folder):
- Overview page: pikalytics_cache/overview_<format>.html
  Example for VGC 2025 Reg H: overview_gen9vgc2025regh.html
- Per-Pokémon page: pikalytics_cache/details_<format>_<pokemon_slug>.html
  Example: details_gen9vgc2025regh_iron_bundle.html

If network is available, this module tries to fetch pages and then caches them.
Otherwise, it uses whatever is already cached.
"""

from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple, Optional
import json
import csv
from datetime import datetime

import pandas as pd
try:
    import requests  # type: ignore
except Exception:  # requests might be unavailable in some envs
    requests = None  # type: ignore

CACHE_DIR = "pikalytics_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    return s


def _save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _cache_json(path: str, payload: Dict[str, object]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _load_cached_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        # allow list payload saved directly
        return {"data": data}
    except Exception:
        return None


def _month_candidates(months_back: int = 24) -> List[str]:
    now = datetime.utcnow()
    year = now.year
    month = now.month
    candidates: List[str] = []
    for _ in range(max(1, months_back)):
        candidates.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month <= 0:
            month = 12
            year -= 1
    # include known historical defaults just in case
    for fallback in ("2025-09", "2025-08", "2025-07", "2024-12"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _fetch_overview_via_api(format_slug: str, html: Optional[str], use_cache: bool) -> List[Tuple[str, float]]:
    if requests is None:
        return []

    cache_file = os.path.join(CACHE_DIR, f"overview_api_{format_slug}.json")
    if use_cache:
        cached = _load_cached_json(cache_file)
        if cached and cached.get("data"):
            try:
                return [
                    (str(entry.get("name", "")).strip(), float(str(entry.get("percent", 0)).replace("%", "")))
                    for entry in cached.get("data", [])
                    if str(entry.get("name", "")).strip()
                ]
            except Exception:
                pass

    key_candidates: List[str] = []
    if html:
        key_candidates.extend(re.findall(r'value="(%s-[0-9]+)"' % re.escape(format_slug), html, re.I))
        m_attr = re.search(r'data-format="(%s-[^"]+)"' % re.escape(format_slug), html, re.I)
        if m_attr:
            key_candidates.append(m_attr.group(1))
    key_candidates.extend([
        f"{format_slug}-1760",
        f"{format_slug}-1630",
        f"{format_slug}-1500",
        format_slug,
    ])
    key_candidates = _unique(key_candidates)

    date_candidates: List[str] = []
    if use_cache and os.path.exists(cache_file):
        cached = _load_cached_json(cache_file)
        if cached:
            hint = str(cached.get("date") or "").strip()
            if hint:
                date_candidates.append(hint)
    date_candidates.extend(_month_candidates())
    date_candidates = _unique(date_candidates)

    for date_str in date_candidates:
        for key in key_candidates:
            url = f"https://www.pikalytics.com/api/l/{date_str}/{key}"
            try:
                resp = requests.get(url, timeout=10)
            except Exception:
                continue
            if resp.status_code != 200:
                continue
            try:
                payload = resp.json()
            except Exception:
                continue
            if not isinstance(payload, list) or not payload:
                continue
            try:
                results: List[Tuple[str, float]] = []
                for entry in payload:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("name", "")).strip()
                    if not name:
                        continue
                    raw_pct = entry.get("percent")
                    try:
                        usage = float(str(raw_pct).replace("%", "").strip())
                    except Exception:
                        usage = 0.0
                    results.append((name, usage))
                if not results:
                    continue
                if use_cache:
                    _cache_json(cache_file, {"date": date_str, "key": key, "data": payload})
                return results
            except Exception:
                continue
    return []


def fetch_overview(format_slug: str = "gen9vgc2025regh", use_cache: bool = True) -> List[Tuple[str, float]]:
    """
    Return list of (pokemon_name, usage_percent) for a given format.
    Prefers cached HTML at overview_<format>.html; attempts fetch if possible.
    """
    cache_file = os.path.join(CACHE_DIR, f"overview_{format_slug}.html")
    html = _read_text(cache_file) if use_cache else None

    if html is None and requests is not None:
        try:
            url = f"https://www.pikalytics.com/pokedex/{format_slug}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
            _save_text(cache_file, html)
        except Exception:
            html = _read_text(cache_file)  # fallback to whatever cached

    if not html:
        return []

    # Parse tables; find the one with columns that contain a Pokemon name and usage percentage
    best: List[Tuple[str, float]] = []
    try:
        tables = pd.read_html(html)
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            has_name = any(("pokemon" in c) or ("pokémon" in c) or ("name" in c) for c in cols)
            has_usage = any(("usage" in c) or ("%" in c) for c in cols)
            if has_name and has_usage:
                # pick name column heuristically
                pcol = None
                for c in t.columns:
                    cl = str(c).strip().lower()
                    if "pokemon" in cl or "pokémon" in cl or cl.startswith("name"):
                        pcol = c
                        break
                if pcol is None:
                    continue
                # pick usage column heuristically
                ucol = None
                for c in t.columns:
                    cl = str(c).strip().lower()
                    if "usage" in cl or "%" in cl:
                        ucol = c
                        break
                if ucol is None:
                    continue
                for _, row in t.iterrows():
                    name = str(row.get(pcol) or "").strip()
                    raw = row.get(ucol)
                    try:
                        usage = float(str(raw).replace('%','').strip())
                    except Exception:
                        continue
                    if name:
                        best.append((name, usage))
                if best:
                    break
    except Exception:
        # ignore and try JSON/regex fallbacks below
        pass

    if best:
        return best

    api_results = _fetch_overview_via_api(format_slug, html, use_cache)
    if api_results:
        return api_results

    # JSON fallback: Next.js __NEXT_DATA__ embedded payload
    try:
        m = re.search(r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S | re.I)
        if m:
            payload = json.loads(m.group(1))

            found: Dict[str, float] = {}

            def walk(obj):
                if isinstance(obj, dict):
                    name = None
                    usage = None
                    # collect candidates
                    for k, v in obj.items():
                        kl = str(k).lower()
                        if kl in ("pokemon", "name", "species") and isinstance(v, str):
                            name = v
                        if "usage" in kl:
                            try:
                                usage = float(str(v).replace('%',''))
                            except Exception:
                                pass
                    if name and usage is not None:
                        # keep max usage seen for this name
                        if name not in found or usage > found[name]:
                            found[name] = usage
                    for v in obj.values():
                        walk(v)
                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)

            walk(payload)
            if found:
                items = list(found.items())
                items.sort(key=lambda x: x[1], reverse=True)
                return items
    except Exception:
        # ignore and try regex fallback below
        pass

    if best:
        return best

    # Regex fallback: scan for links to the format and nearest percentage
    # This is resilient to dynamic markup where tables aren't present.
    try:
        pattern = re.compile(rf"/pokedex/{re.escape(format_slug)}/([a-z0-9\-]+)", re.I)
        seen = set()
        fallback: List[Tuple[str, float]] = []
        for m in pattern.finditer(html):
            slug = m.group(1)
            if slug in seen:
                continue
            seen.add(slug)
            # Grab a window around the match to find a nearby percentage
            start = max(0, m.start() - 300)
            end = min(len(html), m.end() + 300)
            window = html[start:end]
            pm = re.search(r"(\d+\.?\d*)\s*%", window)
            if pm:
                pct = float(pm.group(1))
            else:
                pct = 0.0
            # Derive a readable name from slug
            name = slug.replace('-', ' ').title()
            fallback.append((name, pct))
        # Sort by usage desc
        fallback.sort(key=lambda x: x[1], reverse=True)
        return fallback
    except Exception:
        return []


def fetch_details(pokemon_name: str, format_slug: str = "gen9vgc2025regh", use_cache: bool = True) -> Dict[str, List[Tuple[str, float]]]:
    """
    Return usage dict with keys: 'moves', 'items', 'abilities'.
    Each value is a list of (name, usage_percent).
    Reads cache first; attempts fetch if possible.
    """
    slug = _slugify(pokemon_name)
    cache_file = os.path.join(CACHE_DIR, f"details_{format_slug}_{slug}.html")
    html = _read_text(cache_file) if use_cache else None

    if html is None and requests is not None:
        try:
            url = f"https://www.pikalytics.com/pokedex/{format_slug}/{slug}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
            _save_text(cache_file, html)
        except Exception:
            html = _read_text(cache_file)

    results = {"moves": [], "items": [], "abilities": []}
    if not html:
        return results

    try:
        tables = pd.read_html(html)
    except Exception:
        return results

    # Heuristics: locate tables by column labels containing keywords
    def extract_by_keywords(keywords: List[str]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any(k in " ".join(cols) for k in keywords) and any("usage" in c for c in cols):
                name_col = None
                for c in t.columns:
                    cl = str(c).strip().lower()
                    if any(k[:-1] if k.endswith('s') else k in cl for k in keywords):
                        name_col = c
                        break
                if name_col is None:
                    continue
                ucol = next((c for c in t.columns if "usage" in str(c).strip().lower()), None)
                if ucol is None:
                    continue
                for _, row in t.iterrows():
                    nm = str(row.get(name_col) or "").strip()
                    try:
                        usage = float(str(row.get(ucol)).replace('%','').strip())
                    except Exception:
                        continue
                    if nm:
                        out.append((nm, usage))
                break
        return out

    results["moves"] = extract_by_keywords(["move", "moves"])
    results["items"] = extract_by_keywords(["item", "items"])
    results["abilities"] = extract_by_keywords(["ability", "abilities"])
    return results


def build_compendium(
    format_slug: str = "gen9vgc2025regh",
    min_usage: float = 0.0,
    top_moves: int = 8,
    top_items: int = 6,
    top_abilities: int = 4,
    use_cache: bool = True,
) -> Dict[str, object]:
    """
    Aggregate overview + per-Pokémon details into a single in-memory structure.
    Returns a dict with keys: 'format', 'min_usage', 'count', 'pokemon'.
    pokemon[name] = { 'usage': float, 'moves': [(name, pct)..], 'items': [...], 'abilities': [...] }
    """
    overview = fetch_overview(format_slug, use_cache=use_cache)
    usage_map: Dict[str, float] = {n: u for n, u in overview if isinstance(u, (int, float))}
    names = [n for n, u in overview if (isinstance(u, (int, float)) and u >= min_usage)]
    comp: Dict[str, object] = {
        "format": format_slug,
        "min_usage": float(min_usage),
        "pokemon": {},
    }
    for name in names:
        det = fetch_details(name, format_slug, use_cache=use_cache)
        entry = {
            "usage": float(usage_map.get(name, 0.0)),
            "moves": det.get("moves", [])[:top_moves],
            "items": det.get("items", [])[:top_items],
            "abilities": det.get("abilities", [])[:top_abilities],
        }
        comp["pokemon"][name] = entry
    comp["count"] = len(comp["pokemon"])  # type: ignore[index]
    return comp


def save_compendium(
    compendium: Dict[str, object],
    format_slug: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    """
    Save aggregated compendium JSON to cache folder (or provided path).
    Returns the file path.
    """
    fmt = format_slug or str(compendium.get("format", "gen9vgc2025regh"))
    if out_path is None:
        out_path = os.path.join(CACHE_DIR, f"compendium_{fmt}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compendium, f, ensure_ascii=False, indent=2)
    return out_path


def build_and_save_compendium(
    format_slug: str = "gen9vgc2025regh",
    min_usage: float = 0.5,
    top_moves: int = 8,
    top_items: int = 6,
    top_abilities: int = 4,
    out_path: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    comp = build_compendium(format_slug, min_usage, top_moves, top_items, top_abilities, use_cache=use_cache)
    return save_compendium(comp, format_slug, out_path)


def save_compendium_csv(
    compendium: Dict[str, object],
    out_dir: Optional[str] = None,
) -> List[str]:
    """
    Save compendium into multiple CSV files:
      - compendium_<format>_overview.csv (name, usage)
      - compendium_<format>_moves.csv (name, move, usage)
      - compendium_<format>_items.csv (name, item, usage)
      - compendium_<format>_abilities.csv (name, ability, usage)
    Returns list of file paths.
    """
    fmt = str(compendium.get("format", "gen9vgc2025regh"))
    if out_dir is None:
        out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    paths: List[str] = []
    overview_path = os.path.join(out_dir, f"compendium_{fmt}_overview.csv")
    moves_path = os.path.join(out_dir, f"compendium_{fmt}_moves.csv")
    items_path = os.path.join(out_dir, f"compendium_{fmt}_items.csv")
    abilities_path = os.path.join(out_dir, f"compendium_{fmt}_abilities.csv")

    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pokemon", "usage"])
        for name, entry in sorted(((n, compendium["pokemon"][n]) for n in compendium.get("pokemon", {})), key=lambda x: x[1].get("usage", 0), reverse=True):
            w.writerow([name, entry.get("usage", 0)])
    paths.append(overview_path)

    def dump_list(path: str, key: str, header: List[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for name, entry in compendium.get("pokemon", {}).items():
                for item_name, pct in entry.get(key, []):
                    w.writerow([name, item_name, pct])
        paths.append(path)

    dump_list(moves_path, "moves", ["pokemon", "move", "usage"]) 
    dump_list(items_path, "items", ["pokemon", "item", "usage"]) 
    dump_list(abilities_path, "abilities", ["pokemon", "ability", "usage"]) 

    return paths


def build_and_save_compendium_csv(
    format_slug: str = "gen9vgc2025regh",
    min_usage: float = 0.5,
    top_moves: int = 8,
    top_items: int = 6,
    top_abilities: int = 4,
    out_dir: Optional[str] = None,
    use_cache: bool = True,
) -> List[str]:
    comp = build_compendium(format_slug, min_usage, top_moves, top_items, top_abilities, use_cache=use_cache)
    return save_compendium_csv(comp, out_dir)


def load_compendium_csv(
    format_slug: str = "gen9vgc2025regh",
    base_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load compendium data from CSV files in base_dir (defaults to cache dir).
    Returns the same structure as build_compendium.
    Missing files are tolerated; only available data is returned.
    """
    if base_dir is None:
        base_dir = CACHE_DIR
    overview_path = os.path.join(base_dir, f"compendium_{format_slug}_overview.csv")
    moves_path = os.path.join(base_dir, f"compendium_{format_slug}_moves.csv")
    items_path = os.path.join(base_dir, f"compendium_{format_slug}_items.csv")
    abilities_path = os.path.join(base_dir, f"compendium_{format_slug}_abilities.csv")

    comp: Dict[str, object] = {"format": format_slug, "pokemon": {}, "min_usage": 0.0}

    # Overview
    if os.path.exists(overview_path):
        with open(overview_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("pokemon") or row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    usage = float(str(row.get("usage") or 0))
                except Exception:
                    usage = 0.0
                comp["pokemon"].setdefault(name, {"usage": usage, "moves": [], "items": [], "abilities": []})
                comp["pokemon"][name]["usage"] = usage  # type: ignore[index]

    # Helper to load list csvs
    def load_list(path: str, value_key: str, dest_key: str):
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("pokemon") or row.get("name") or "").strip()
                item = (row.get(value_key) or row.get(dest_key) or "").strip()
                if not name or not item:
                    continue
                try:
                    pct = float(str(row.get("usage") or 0))
                except Exception:
                    pct = 0.0
                comp["pokemon"].setdefault(name, {"usage": 0.0, "moves": [], "items": [], "abilities": []})
                comp["pokemon"][name].setdefault(dest_key, [])  # type: ignore[index]
                comp["pokemon"][name][dest_key].append((item, pct))  # type: ignore[index]

    load_list(moves_path, "move", "moves")
    load_list(items_path, "item", "items")
    load_list(abilities_path, "ability", "abilities")

    comp["count"] = len(comp["pokemon"])  # type: ignore[index]
    return comp


 
def save_compendium_single_csv(
    compendium: Dict[str, object],
    out_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Save the entire compendium into a single CSV file with schema:
      type,pokemon,value,usage
    where type in {overview,move,item,ability} and value is the move/item/ability
    (for overview rows, value is empty).
    Returns the file path.
    """
    fmt = str(compendium.get("format", "gen9vgc2025regh"))
    if out_dir is None:
        out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename or f"compendium_{fmt}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "pokemon", "value", "usage"])  # header
        # Overview rows
        for name, entry in compendium.get("pokemon", {}).items():
            usage = float(entry.get("usage", 0.0))
            w.writerow(["overview", name, "", usage])
        # Helper to dump lists
        def dump_list(kind: str, key: str):
            for name, entry in compendium.get("pokemon", {}).items():
                for v, pct in entry.get(key, []):
                    w.writerow([kind, name, v, float(pct)])
        dump_list("move", "moves")
        dump_list("item", "items")
        dump_list("ability", "abilities")
    return out_path


def build_and_save_compendium_single_csv(
    format_slug: str = "gen9vgc2025regh",
    min_usage: float = 0.5,
    top_moves: int = 8,
    top_items: int = 6,
    top_abilities: int = 4,
    out_dir: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    comp = build_compendium(format_slug, min_usage, top_moves, top_items, top_abilities, use_cache=use_cache)
    return save_compendium_single_csv(comp, out_dir)


def load_compendium_xml(
    format_slug: str = "gen9vgc2025regh",
    base_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load compendium from a single XML file with schema:
      <compendium format="...">
        <pokemon name="..." usage="...">
          <abilities><ability name="..." usage="..."/></abilities>
          <items><item name="..." usage="..."/></items>
          <moves><move name="..." usage="..."/></moves>
        </pokemon>
      </compendium>
    Returns the same dict structure as build_compendium.
    """
    if base_dir is None:
        base_dir = CACHE_DIR
    path = os.path.join(base_dir, filename or f"compendium_{format_slug}.xml")
    comp: Dict[str, object] = {"format": format_slug, "pokemon": {}, "min_usage": 0.0}
    if not os.path.exists(path):
        return comp
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        fmt = root.attrib.get("format") or format_slug
        comp["format"] = fmt  # type: ignore[index]
        for pnode in root.findall("pokemon"):
            name = (pnode.attrib.get("name") or "").strip()
            if not name:
                continue
            try:
                usage = float(pnode.attrib.get("usage") or 0)
            except Exception:
                usage = 0.0
            entry = {"usage": usage, "abilities": [], "items": [], "moves": []}
            an = pnode.find("abilities")
            if an is not None:
                for a in an.findall("ability"):
                    nm = (a.attrib.get("name") or "").strip()
                    try:
                        pct = float(a.attrib.get("usage") or 0)
                    except Exception:
                        pct = 0.0
                    if nm:
                        entry["abilities"].append((nm, pct))  # type: ignore[index]
            in_ = pnode.find("items")
            if in_ is not None:
                for it in in_.findall("item"):
                    nm = (it.attrib.get("name") or "").strip()
                    try:
                        pct = float(it.attrib.get("usage") or 0)
                    except Exception:
                        pct = 0.0
                    if nm:
                        entry["items"].append((nm, pct))  # type: ignore[index]
            mn = pnode.find("moves")
            if mn is not None:
                for mv in mn.findall("move"):
                    nm = (mv.attrib.get("name") or "").strip()
                    try:
                        pct = float(mv.attrib.get("usage") or 0)
                    except Exception:
                        pct = 0.0
                    if nm:
                        entry["moves"].append((nm, pct))  # type: ignore[index]
            comp["pokemon"][name] = entry  # type: ignore[index]
        comp["count"] = len(comp["pokemon"])  # type: ignore[index]
        return comp
    except Exception:
        return comp

def load_compendium_single_csv(
    format_slug: str = "gen9vgc2025regh",
    base_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load a single CSV compendium created by save_compendium_single_csv.
    Returns the compendium dict structure.
    """
    if base_dir is None:
        base_dir = CACHE_DIR
    path = os.path.join(base_dir, filename or f"compendium_{format_slug}.csv")
    comp: Dict[str, object] = {"format": format_slug, "pokemon": {}, "min_usage": 0.0}
    if not os.path.exists(path):
        return comp
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            kind = (row.get("type") or "").strip().lower()
            name = (row.get("pokemon") or "").strip()
            value = (row.get("value") or "").strip()
            try:
                pct = float(str(row.get("usage") or 0))
            except Exception:
                pct = 0.0
            if not name:
                continue
            comp["pokemon"].setdefault(name, {"usage": 0.0, "moves": [], "items": [], "abilities": []})
            if kind == "overview":
                comp["pokemon"][name]["usage"] = pct  # type: ignore[index]
            elif kind == "move":
                comp["pokemon"][name]["moves"].append((value, pct))  # type: ignore[index]
            elif kind == "item":
                comp["pokemon"][name]["items"].append((value, pct))  # type: ignore[index]
            elif kind == "ability":
                comp["pokemon"][name]["abilities"].append((value, pct))  # type: ignore[index]
    comp["count"] = len(comp["pokemon"])  # type: ignore[index]
    return comp
