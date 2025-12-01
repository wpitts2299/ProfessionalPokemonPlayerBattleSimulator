"""
Team Builder: balanced and random team generation with simple synergy scoring.

Exports:
- generate_balanced_team(stats_df, moves_df, abilities_df, n=4, item_style="balanced")
- generate_random_team(stats_df, moves_df, abilities_df, n=4, item_style="balanced")
"""

import os
import random
from itertools import combinations
from typing import Iterable, List, Dict, Optional, Set, Tuple

from battle_ai import TYPE_EFFECTIVENESS, _norm_type
from data_loader import (
    create_pokemon_from_name,
    Move,
    Pokemon,
)
from data_loader import _lookup_move_in_db  # use CSV-backed move details
from data_loader import _select_moves_with_setup  # ensure AI rosters include setup options
from pikalytics_util import (
    CACHE_DIR,
    fetch_details,
    fetch_overview,
    load_compendium_csv,
    load_compendium_single_csv,
)

 
AGGRESSIVE_ITEMS = ["Choice Band", "Choice Specs", "Life Orb", "Expert Belt", "Choice Scarf"]
DEFENSIVE_ITEMS = ["Leftovers", "Assault Vest", "Rocky Helmet", "Sitrus Berry"]
BALANCED_ITEMS = ["Leftovers", "Expert Belt", "Life Orb", "Choice Scarf", "Sitrus Berry"]

AI_BANNED_MOVES = {"protect"}


def pick_item(style: str = "balanced") -> Optional[str]:
    s = (style or "balanced").lower()
    if s == "aggressive":
        pool = AGGRESSIVE_ITEMS
    elif s == "defensive":
        pool = DEFENSIVE_ITEMS
    else:
        pool = BALANCED_ITEMS
    return random.choice(pool) if pool else None


 
COVERAGE_PRIORITIES = [
    "Water",
    "Fire",
    "Grass",
    "Electric",
    "Ground",
    "Ice",
    "Fairy",
    "Dragon",
    "Fighting",
]

UTILITY_KEYWORDS = [
    "protect",
    "substitute",
    "roost",
    "rest",
    "synthesis",
    "recover",
    "wish",
    "heal",
    "swords",
    "dragon dance",
    "nasty plot",
    "calm mind",
    "spikes",
    "stealth rock",
    "toxic",
    "bulk",
    "shell smash",
]

MAX_MOVES_PER_MON = 4

Team = List[Pokemon]

DEFAULT_USAGE_FORMAT = "gen9vgc2025regh"
DEFAULT_TEAM_SIZE = 4
ALL_TYPES = sorted(TYPE_EFFECTIVENESS.keys())
STAT_COLUMNS = [
    "hp",
    "attack",
    "defense",
    "special_attack",
    "sp_attack",
    "special_defense",
    "sp_defense",
    "speed",
]


def _clean_string(value) -> str:
    if value is None:
        return ""
    try:
        text = str(value).strip()
    except Exception:
        return ""
    if not text or text.lower() == "nan":
        return ""
    return text


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return default
            value = txt
        return float(value)
    except Exception:
        return default


def _extract_name(record: Dict[str, object]) -> str:
    for col in ("pokemon", "name", "original_name"):
        if col in record:
            name = _clean_string(record[col])
            if name:
                return name
    return ""


def _extract_types(record: Dict[str, object]) -> List[str]:
    types: List[str] = []
    primary_candidates = ["type1", "primary_type", "type"]
    secondary_candidates = ["type2", "secondary_type", "type_2"]

    for col in primary_candidates:
        if col in record:
            t = _clean_string(record[col]).title()
            if t:
                types.append(t)
                break
    for col in secondary_candidates:
        if col in record:
            t = _clean_string(record[col]).title()
            if t:
                types.append(t)
                break
    unique: List[str] = []
    for t in types:
        if t and t not in unique:
            unique.append(t)
    return unique


def _sum_stats(record: Dict[str, object]) -> float:
    total = 0.0
    seen: Set[str] = set()
    for col in STAT_COLUMNS:
        if col in record and col not in seen:
            total += _coerce_float(record.get(col), 0.0)
            seen.add(col)
    return total


def _load_usage_pairs(format_slug: str = DEFAULT_USAGE_FORMAT, limit: int = 300) -> List[Tuple[str, float]]:
    fmt = format_slug or DEFAULT_USAGE_FORMAT
    try:
        single_csv = os.path.join(CACHE_DIR, f"compendium_{fmt}.csv")
        overview_csv = os.path.join(CACHE_DIR, f"compendium_{fmt}_overview.csv")
        comp = None
        if os.path.exists(single_csv):
            comp = load_compendium_single_csv(fmt)
        elif os.path.exists(overview_csv):
            comp = load_compendium_csv(fmt)
        if comp and comp.get("pokemon"):
            rows = []
            for name, payload in comp["pokemon"].items():  # type: ignore[index]
                nm = _clean_string(name)
                if not nm:
                    continue
                try:
                    usage = float(payload.get("usage", 0.0))  # type: ignore[arg-type]
                except Exception:
                    usage = 0.0
                rows.append((nm, usage))
            rows.sort(key=lambda x: x[1], reverse=True)
            if limit:
                rows = rows[:limit]
            return rows
    except Exception:
        pass

    overview = fetch_overview(fmt)
    cleaned: List[Tuple[str, float]] = []
    for name, usage in overview:
        nm = _clean_string(name)
        if not nm:
            continue
        try:
            pct = float(usage)
        except Exception:
            pct = _coerce_float(usage, 0.0)
        cleaned.append((nm, pct))
    cleaned.sort(key=lambda x: x[1], reverse=True)
    if limit:
        cleaned = cleaned[:limit]
    return cleaned


def _prepare_usage_context(
    stats_df,
    usage_pairs: List[Tuple[str, float]],
    allowed_names: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    usage_map = {name.lower(): float(usage) for name, usage in usage_pairs if _clean_string(name)}
    try:
        records = stats_df.to_dict(orient="records")
    except Exception:
        records = []
    entries_map: Dict[str, Dict[str, object]] = {}
    for record in records:
        name = _extract_name(record)
        if not name:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        types = _extract_types(record)
        if not types:
            continue
        total = _coerce_float(record.get("total"), 0.0)
        if total <= 0:
            total = _sum_stats(record)
        entry = {
            "name": name,
            "usage": usage_map.get(name.lower(), 0.0),
            "total": total,
            "types": types,
        }
        existing = entries_map.get(name)
        if existing is None or (entry["usage"], entry["total"]) > (existing["usage"], existing["total"]):
            entries_map[name] = entry

    entries = list(entries_map.values())
    entries.sort(key=lambda e: (e["usage"], e["total"]), reverse=True)
    base_sorted = sorted(entries, key=lambda e: (e["total"], e["usage"]), reverse=True)

    type_index: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        for t in entry.get("types", []):
            if not t:
                continue
            type_index.setdefault(t, []).append(entry)
    for tp_entries in type_index.values():
        tp_entries.sort(key=lambda e: (e["usage"], e["total"]), reverse=True)

    return entries, base_sorted, type_index


def _weaknesses_for_types(types: List[str]) -> List[str]:
    cleaned = [_norm_type(t) for t in types if _clean_string(t)]
    if not cleaned:
        return []
    weaknesses: List[str] = []
    for atk in ALL_TYPES:
        atk_norm = _norm_type(atk)
        mult = 1.0
        for defend in cleaned:
            mult *= TYPE_EFFECTIVENESS.get(atk_norm, {}).get(defend, 1.0)
        if mult > 1.0 and atk_norm not in weaknesses:
            weaknesses.append(atk_norm)
    return weaknesses


 
def _moves_from_moves_df_for_pokemon(name: str, moves_df, prefer_setup: bool = False) -> List[Move]:
    # If moves_df is a global move database without per-Pokémon mapping, return empty
    if "pokemon" not in moves_df.columns:
        return []
    subset = moves_df[moves_df["pokemon"].str.lower() == name.lower()]
    if subset.empty:
        return []

    subset = subset.copy()
    # Ensure numeric columns exist with defaults
    if "power" in subset.columns:
        subset["power"] = subset["power"].apply(
            lambda x: float(x) if (x is not None and str(x).strip() != "") else 0.0
        )
    else:
        subset["power"] = 0.0
    if "accuracy" in subset.columns:
        subset["accuracy"] = subset["accuracy"].apply(
            lambda x: float(x) if (x is not None and str(x).strip() != "") else 100.0
        )
    else:
        subset["accuracy"] = 100.0

    def score_row(r):
        power = float(r.get("power", 0.0) or 0.0)
        acc = float(r.get("accuracy", 100.0) or 100.0)
        name_lower = str(r.get("move") or r.get("name") or "").lower()
        util_bonus = 0.0
        for kw in UTILITY_KEYWORDS:
            if kw in name_lower:
                util_bonus += 10.0
        if prefer_setup and util_bonus > 0:
            util_bonus += 50.0
        return (power, acc, util_bonus)

    rows = list(subset.iterrows())
    rows.sort(key=lambda x: score_row(x[1]), reverse=True)

    moves: List[Move] = []
    for _, row in rows:
        name_col = row.get("move") or row.get("name")
        mtype = row.get("type") or "Normal"
        # Defensive numeric coercion
        try:
            power = int(float(row.get("power") or 0))
        except Exception:
            power = 0
        try:
            accuracy = int(float(row.get("accuracy") or 100))
        except Exception:
            accuracy = 100
        pp = int(row.get("pp") or 10) if "pp" in row.index else 10
        category = row.get("category") or ("special" if row.get("is_special", False) else "physical")
        moves.append(Move(name=name_col, power=power, mtype=mtype, accuracy=accuracy, pp=pp, category=category))
    return moves


def _choose_moves_for_pokemon_advanced(
    pkm: Pokemon,
    moves_df,
    abilities_df,
    use_pikalytics: bool = False,
    banned_moves: Optional[Set[str]] = None,
) -> List[Move]:
    name = pkm.name
    banned_lc = {str(m or "").strip().lower() for m in (banned_moves or set()) if str(m or "").strip()}

    def _filter_banned(moves: Iterable[Move]) -> List[Move]:
        if not banned_lc:
            return [mv for mv in moves if isinstance(mv, Move)]
        filtered: List[Move] = []
        for mv in moves:
            if not isinstance(mv, Move):
                continue
            if str(getattr(mv, "name", "") or "").strip().lower() in banned_lc:
                continue
            filtered.append(mv)
        return filtered

    candidate_moves = _filter_banned(_moves_from_moves_df_for_pokemon(name, moves_df, prefer_setup=False))

    # Strong preference: Pikalytics-tracked moves in reported order
    try:
        det = fetch_details(name)
        pika_names = [nm for nm, _ in det.get("moves", [])] if isinstance(det, dict) else []
    except Exception:
        pika_names = []

    if pika_names:
        # Use Pikalytics move frequency first, then CSV candidates to fill gaps.
        ordered: List[Move] = []
        for nm in pika_names:
            if banned_lc and str(nm or "").strip().lower() in banned_lc:
                continue
            mv = _lookup_move_in_db(nm, moves_df)
            if mv:
                ordered.append(mv)
        ordered.extend(candidate_moves)
        selected = _select_moves_with_setup(_filter_banned(ordered), max_moves=MAX_MOVES_PER_MON)
        if selected:
            return selected

    if not candidate_moves:
        fallback_existing = _filter_banned(getattr(pkm, "moves", []) or [])
        if fallback_existing:
            return fallback_existing[:MAX_MOVES_PER_MON]
        return [Move("Tackle", 40, "Normal", 100, 35, "physical")]

    # Fall back to CSV data entirely when Pikalytics is unavailable.
    ordered = list(candidate_moves)
    selected = _select_moves_with_setup(_filter_banned(ordered), max_moves=MAX_MOVES_PER_MON)
    if selected:
        return selected

    fallback = _filter_banned(getattr(pkm, "moves", []) or [])
    if fallback:
        return fallback[:MAX_MOVES_PER_MON]
    return [Move("Tackle", 40, "Normal", 100, 35, "physical")]


 
def _team_coverage_score(team: Team) -> float:
    covered: Set[str] = set()
    for p in team:
        for mv in p.moves:
            if mv.power <= 0:
                continue
            covered.add(mv.type)
    coverage_hits = sum(1 for t in COVERAGE_PRIORITIES if t in covered)
    return coverage_hits / max(1, len(COVERAGE_PRIORITIES))


def _team_type_diversity_score(team: Team) -> float:
    types = []
    for p in team:
        for t in p.type:
            types.append(t)
    unique_types = len(set(types))
    return unique_types / (len(team) * 2)


def _team_bulk_score(team: Team) -> float:
    bulks = [p.max_hp + getattr(p, "defense", 0) + getattr(p, "special_defense", 0) for p in team]
    if not bulks:
        return 0.0
    top = max(bulks)
    avg = sum(bulks) / len(bulks)
    score = (top / max(1.0, avg)) - 0.5
    return max(0.0, min(1.0, score))


def _team_physical_special_balance(team: Team) -> float:
    phys = 0
    spec = 0
    for p in team:
        for m in p.moves:
            if getattr(m, "is_special", False) or str(getattr(m, "category", "")).lower() == "special":
                spec += 1
            else:
                phys += 1
    total = phys + spec
    if total == 0:
        return 0.0
    frac = phys / total
    return 1.0 - min(1.0, abs(frac - 0.5) * 2.0)


def _weather_synergy_score(team: Team) -> float:
    weather_map = {"Drizzle": "Rain", "Drought": "Sun", "Sand Stream": "Sand", "Snow Warning": "Snow"}
    setter = None
    for p in team:
        if p.ability in weather_map:
            setter = (p, weather_map[p.ability])
            break
    if not setter:
        return 0.0
    _, wt = setter
    benefit_types = {
        "Rain": ["Water", "Electric"],
        "Sun": ["Fire", "Grass"],
        "Sand": ["Rock", "Ground", "Steel"],
        "Snow": ["Ice"],
    }.get(wt, [])
    count = sum(1 for p in team if any(t in benefit_types for t in p.type))
    return count / max(1, len(team))


def _team_synergy_score(team: Team) -> float:
    coverage = _team_coverage_score(team)
    diversity = _team_type_diversity_score(team)
    bulk = _team_bulk_score(team)
    balance = _team_physical_special_balance(team)
    weather = _weather_synergy_score(team)
    return (coverage * 4.0) + (diversity * 1.5) + (bulk * 1.2) + (balance * 1.0) + (weather * 1.5)


class _StructuredTeamBuilder:
    """Build a roster following the step-by-step selection algorithm."""

    def __init__(
        self,
        stats_df,
        moves_df,
        abilities_df,
        usage_pairs: List[Tuple[str, float]],
        team_size: int,
        item_style: str,
        use_pikalytics: bool,
        allowed_names: Optional[Set[str]] = None,
        banned_moves: Optional[Set[str]] = None,
    ) -> None:
        self.stats_df = stats_df
        self.moves_df = moves_df
        self.abilities_df = abilities_df
        self.team_size = max(1, int(team_size or DEFAULT_TEAM_SIZE))
        self.core_target = max(1, min(DEFAULT_TEAM_SIZE, self.team_size))
        self.item_style = item_style
        self.use_pikalytics = use_pikalytics
        self.banned_moves = banned_moves
        self.team: Team = []
        self.used_names: Set[str] = set()
        self.looked_names: Set[str] = set()

        entries, base_entries, type_index = _prepare_usage_context(stats_df, usage_pairs, allowed_names)
        self.usage_entries = entries
        self.base_entries = base_entries
        self.type_index = type_index
        self.available_types = [t for t, lst in type_index.items() if lst]

    def build(self) -> Team:
        if not self.usage_entries:
            return []
        if not self._pick_initial_member():
            return []
        first = self.team[0]
        weaknesses = _weaknesses_for_types(getattr(first, "type", []))

        self._try_three_type_branch(weaknesses)
        if not self._core_complete():
            self._try_two_type_branch(weaknesses)
        if not self._core_complete():
            self._try_single_type_branch(weaknesses)
        if not self._core_complete():
            self._fill_core_with_usage()

        self._fill_remaining_slots()
        return self.team

    # ---------------- Selection helpers ----------------
    def _pick_initial_member(self) -> bool:
        pool = [entry for entry in self.usage_entries if entry["name"] not in self.used_names]
        pool = pool[:50] if len(pool) >= 50 else pool[:]
        if not pool:
            return False
        self._mark_looked(pool)
        random.shuffle(pool)
        for entry in pool:
            if self._add_entry(entry):
                return True
        return False

    def _add_entry(self, entry: Dict[str, object]) -> Optional[Pokemon]:
        name = str(entry.get("name"))
        if not name or name in self.used_names:
            return None
        pokemon = self._instantiate_pokemon(name)
        if not pokemon:
            return None
        self.team.append(pokemon)
        self.used_names.add(pokemon.name)
        self.looked_names.add(pokemon.name)
        return pokemon

    def _instantiate_pokemon(self, name: str) -> Optional[Pokemon]:
        try:
            pk = create_pokemon_from_name(name, self.stats_df, self.moves_df, self.abilities_df, preferred_item=pick_item(self.item_style))
            if len(getattr(pk, "moves", [])) != MAX_MOVES_PER_MON or self.banned_moves:
                pk.moves = _choose_moves_for_pokemon_advanced(
                    pk,
                    self.moves_df,
                    self.abilities_df,
                    self.use_pikalytics,
                    banned_moves=self.banned_moves,
                )
            return pk
        except Exception:
            return None

    def _mark_looked(self, entries: List[Dict[str, object]]) -> None:
        for entry in entries:
            nm = str(entry.get("name") or "")
            if nm:
                self.looked_names.add(nm)

    def _core_complete(self) -> bool:
        return len(self.team) >= self.core_target

    def _type_is_safe(self, def_type: str, threat_types: List[str]) -> bool:
        if not threat_types:
            return True
        defended = _norm_type(def_type)
        for atk in threat_types:
            atk_norm = _norm_type(atk)
            if TYPE_EFFECTIVENESS.get(atk_norm, {}).get(defended, 1.0) > 1.0:
                return False
        return True

    def _types_safe_against(self, threat_types: List[str]) -> List[str]:
        pool = self.available_types or ALL_TYPES
        safe = [t for t in pool if self._type_is_safe(t, threat_types)]
        return safe

    def _types_pairwise_safe(self, type_combo: Tuple[str, ...]) -> bool:
        for a, b in combinations(type_combo, 2):
            a_norm = _norm_type(a)
            b_norm = _norm_type(b)
            if TYPE_EFFECTIVENESS.get(a_norm, {}).get(b_norm, 1.0) > 1.0:
                return False
            if TYPE_EFFECTIVENESS.get(b_norm, {}).get(a_norm, 1.0) > 1.0:
                return False
        return True

    def _type_candidates(self, type_name: str) -> List[Dict[str, object]]:
        entries = self.type_index.get(type_name, [])
        return [entry for entry in entries if entry["name"] not in self.used_names]

    def _add_random_from_type(self, type_name: str, limit: int) -> Optional[Pokemon]:
        candidates = self._type_candidates(type_name)
        if not candidates:
            return None
        pool = candidates[:limit]
        if not pool:
            return None
        self._mark_looked(pool)
        random.shuffle(pool)
        for entry in pool:
            member = self._add_entry(entry)
            if member:
                return member
        return None

    def _add_from_usage(self, limit: int, require_unseen: bool) -> Optional[Pokemon]:
        pool = []
        for entry in self.usage_entries:
            name = entry["name"]
            if name in self.used_names:
                continue
            if require_unseen and name in self.looked_names:
                continue
            pool.append(entry)
            if len(pool) >= limit:
                break
        if not pool:
            return None
        self._mark_looked(pool)
        random.shuffle(pool)
        for entry in pool:
            member = self._add_entry(entry)
            if member:
                return member
        return None

    def _add_from_base_stats(self, limit: int, require_unseen: bool) -> Optional[Pokemon]:
        pool = []
        for entry in self.base_entries:
            name = entry["name"]
            if name in self.used_names:
                continue
            if require_unseen and name in self.looked_names:
                continue
            pool.append(entry)
            if len(pool) >= limit:
                break
        if not pool:
            return None
        self._mark_looked(pool)
        random.shuffle(pool)
        for entry in pool:
            member = self._add_entry(entry)
            if member:
                return member
        return None

    # ---------------- Selection branches ----------------
    def _try_three_type_branch(self, weaknesses: List[str]) -> None:
        safe_types = self._types_safe_against(weaknesses)
        combos = [combo for combo in combinations(safe_types, 3) if self._types_pairwise_safe(combo)]
        random.shuffle(combos)
        if not combos:
            return
        chosen = combos[0]
        for tp in chosen:
            if self._core_complete():
                break
            self._add_random_from_type(tp, limit=5)

    def _try_two_type_branch(self, weaknesses: List[str]) -> None:
        safe_types = self._types_safe_against(weaknesses)
        if len(safe_types) < 2:
            return
        random.shuffle(safe_types)
        chosen = safe_types[:2]
        for tp in chosen:
            if self._core_complete():
                break
            self._add_random_from_type(tp, limit=10)
        if not self._core_complete():
            self._add_from_usage(limit=10, require_unseen=True)

    def _try_single_type_branch(self, weaknesses: List[str]) -> None:
        safe_types = self._types_safe_against(weaknesses)
        pool = safe_types if safe_types else (self.available_types or ALL_TYPES)
        if pool:
            tp = random.choice(pool)
            if not self._core_complete():
                self._add_random_from_type(tp, limit=15)
        if not self._core_complete() and len(self.team) >= 2:
            second = self.team[1]
            sec_safe = self._types_safe_against(_weaknesses_for_types(second.type))
            random.shuffle(sec_safe)
            targets = sec_safe[:2] if len(sec_safe) >= 2 else sec_safe[:1]
            for tp in targets:
                if self._core_complete():
                    break
                self._add_random_from_type(tp, limit=5)
        if not self._core_complete():
            self._add_from_usage(limit=15, require_unseen=True)
        if not self._core_complete():
            self._add_from_base_stats(limit=10, require_unseen=True)

    def _fill_core_with_usage(self) -> None:
        while not self._core_complete():
            member = self._add_from_usage(limit=15, require_unseen=True)
            if not member:
                break

    def _fill_remaining_slots(self) -> None:
        while len(self.team) < self.team_size:
            member = self._add_from_usage(limit=25, require_unseen=False)
            if member:
                continue
            member = self._add_from_base_stats(limit=15, require_unseen=False)
            if member:
                continue
            break

 
def generate_random_team(
    stats_df,
    moves_df,
    abilities_df,
    n: int = DEFAULT_TEAM_SIZE,
    use_pikalytics: bool = False,
    item_style: str = "balanced",
    allowed_names: Optional[Set[str]] = None,
    banned_moves: Optional[Set[str]] = None,
) -> Team:
    names = list(stats_df["pokemon"].unique())
    if allowed_names is not None:
        names = [nm for nm in names if nm in allowed_names]
    random.shuffle(names)
    team: Team = []
    for name in names:
        if len(team) >= n:
            break
        try:
            p = create_pokemon_from_name(name, stats_df, moves_df, abilities_df, preferred_item=pick_item(item_style))
            p.moves = _choose_moves_for_pokemon_advanced(
                p,
                moves_df,
                abilities_df,
                use_pikalytics,
                banned_moves=banned_moves,
            )
            team.append(p)
        except Exception:
            continue
    return team


def generate_balanced_team(
    stats_df,
    moves_df,
    abilities_df,
    n: int = DEFAULT_TEAM_SIZE,
    candidate_pool_size: int = 60,
    use_pikalytics: bool = False,
    item_style: str = "balanced",
    allowed_names: Optional[Set[str]] = None,
    banned_moves: Optional[Set[str]] = None,
    exclude_names: Optional[Set[str]] = None,
) -> Team:
    all_names = list(stats_df["pokemon"].unique())
    if allowed_names is not None:
        all_names = [nm for nm in all_names if nm in allowed_names]
    if exclude_names:
        all_names = [nm for nm in all_names if nm not in exclude_names]
        if not all_names:
            # If exclusions wipe the pool, relax the exclusion to keep teams buildable.
            all_names = list(stats_df["pokemon"].unique())
            if allowed_names is not None:
                all_names = [nm for nm in all_names if nm in allowed_names]
    random.shuffle(all_names)
    pool_names = all_names[: min(candidate_pool_size, len(all_names))]

    candidates: List[Pokemon] = []
    for name in pool_names:
        if exclude_names and name in exclude_names:
            continue
        try:
            p = create_pokemon_from_name(name, stats_df, moves_df, abilities_df, preferred_item=pick_item(item_style))
            p.moves = _choose_moves_for_pokemon_advanced(
                p,
                moves_df,
                abilities_df,
                use_pikalytics,
                banned_moves=banned_moves,
            )
            candidates.append(p)
        except Exception:
            continue

    # If no candidates created, fallback to random
    if not candidates:
        return generate_random_team(
            stats_df,
            moves_df,
            abilities_df,
            n=n,
            use_pikalytics=use_pikalytics,
            item_style=item_style,
            allowed_names=allowed_names,
            banned_moves=banned_moves,
        )

    # Greedy selection maximizing synergy
    team: Team = []
    remaining = candidates[:]
    remaining.sort(key=lambda x: (x.max_hp + getattr(x, "defense", 0) + getattr(x, "special_defense", 0)), reverse=True)

    while len(team) < n and remaining:
        best_candidate = None
        best_score = float("-inf")
        # Evaluate a shuffled subset so repeated builds don't anchor on the same mon (e.g., Tyranitar).
        sample = random.sample(remaining, min(len(remaining), 12))
        for cand in sample:
            if cand.name in [p.name for p in team]:
                continue
            trial = team + [cand]
            score = _team_synergy_score(trial)
            # small penalty for type over-representation
            type_counts: Dict[str, int] = {}
            for p in trial:
                for t in p.type:
                    type_counts[t] = type_counts.get(t, 0) + 1
            penalty = sum(max(0, c - 2) for c in type_counts.values()) * 0.2
            score -= penalty
            score *= (1.0 + (random.random() - 0.5) * 0.3)
            if score > best_score:
                best_score = score
                best_candidate = cand
        if best_candidate is not None:
            team.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            team.append(remaining.pop(0))

    # Ensure 4 moves per mon
    for i, p in enumerate(team):
        if len(p.moves) != MAX_MOVES_PER_MON or banned_moves:
            p.moves = _choose_moves_for_pokemon_advanced(
                p,
                moves_df,
                abilities_df,
                use_pikalytics,
                banned_moves=banned_moves,
            )

    return team


def generate_flowchart_ai_team(
    stats_df,
    moves_df,
    abilities_df,
    n: int = DEFAULT_TEAM_SIZE,
    format_slug: str = DEFAULT_USAGE_FORMAT,
    item_style: str = "aggressive",
    use_pikalytics: bool = False,
    allowed_names: Optional[Set[str]] = None,
    banned_moves: Optional[Set[str]] = None,
) -> Team:
    """Build the opponent roster using the structured selection algorithm."""

    usage_pairs = _load_usage_pairs(format_slug)
    effective_banned = banned_moves or AI_BANNED_MOVES
    if not usage_pairs:
        return generate_balanced_team(
            stats_df,
            moves_df,
            abilities_df,
            n=n,
            item_style=item_style,
            use_pikalytics=use_pikalytics,
            allowed_names=allowed_names,
            banned_moves=effective_banned,
        )

    builder = _StructuredTeamBuilder(
        stats_df,
        moves_df,
        abilities_df,
        usage_pairs,
        team_size=n,
        item_style=item_style,
        use_pikalytics=use_pikalytics,
        allowed_names=allowed_names,
        banned_moves=effective_banned,
    )
    team = builder.build()

    if len(team) < n:
        fallback = generate_balanced_team(
            stats_df,
            moves_df,
            abilities_df,
            n=n,
            item_style=item_style,
            use_pikalytics=use_pikalytics,
            allowed_names=allowed_names,
            banned_moves=effective_banned,
        )
        existing = {p.name for p in team}
        for member in fallback:
            if member.name in existing:
                continue
            team.append(member)
            existing.add(member.name)
            if len(team) >= n:
                break

    return team[:n]


def build_team_from_prompt(
    prompt: str, stats_df, moves_df, abilities_df, n: int = DEFAULT_TEAM_SIZE, use_pikalytics: bool = False, item_style: str = "balanced"
) -> Team:
    prompt_lower = prompt.lower()
    detected = []
    names = list(stats_df["pokemon"].unique())
    for name in names:
        if name.lower() in prompt_lower:
            detected.append(name)
            if len(detected) >= n:
                break

    team: Team = []
    for name in detected:
        try:
            p = create_pokemon_from_name(name, stats_df, moves_df, abilities_df, preferred_item=pick_item(item_style))
            p.moves = _choose_moves_for_pokemon_advanced(p, moves_df, abilities_df, use_pikalytics)
            team.append(p)
        except Exception:
            continue

    # Fill remaining with balanced selection
    if len(team) < n:
        additional = generate_balanced_team(
            stats_df, moves_df, abilities_df, n=n, use_pikalytics=use_pikalytics, item_style=item_style
        )
        for p in additional:
            if p.name not in [q.name for q in team] and len(team) < n:
                team.append(p)

    # Final pass to ensure 4 moves
    for i, p in enumerate(team):
        if len(p.moves) != MAX_MOVES_PER_MON:
            p.moves = _choose_moves_for_pokemon_advanced(p, moves_df, abilities_df, use_pikalytics)

    return team


__all__ = [
    "generate_balanced_team",
    "generate_random_team",
    "generate_flowchart_ai_team",
    "build_team_from_prompt",
    "pick_item",
]
