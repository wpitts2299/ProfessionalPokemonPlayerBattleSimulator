"""
Team Builder: balanced and random team generation with simple synergy scoring.

Exports:
- generate_balanced_team(stats_df, moves_df, abilities_df, n=6, item_style="balanced")
- generate_random_team(stats_df, moves_df, abilities_df, n=6, item_style="balanced")
"""

import random
from typing import List, Dict, Optional, Set

from data_loader import (
    create_pokemon_from_name,
    fetch_pikalytics,
    Move,
    Pokemon,
)

 
AGGRESSIVE_ITEMS = ["Choice Band", "Choice Specs", "Life Orb", "Expert Belt", "Choice Scarf"]
DEFENSIVE_ITEMS = ["Leftovers", "Assault Vest", "Rocky Helmet", "Sitrus Berry"]
BALANCED_ITEMS = ["Leftovers", "Expert Belt", "Life Orb", "Choice Scarf", "Sitrus Berry"]


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


 
def _moves_from_moves_df_for_pokemon(name: str, moves_df, prefer_setup: bool = False) -> List[Move]:
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
    pkm: Pokemon, moves_df, abilities_df, use_pikalytics: bool = False
) -> List[Move]:
    name = pkm.name
    candidate_moves = _moves_from_moves_df_for_pokemon(name, moves_df, prefer_setup=False)

    # Attempt to use pikalytics usage ordering if available
    pikalytics_moves: List[str] = []
    if use_pikalytics:
        try:
            data = fetch_pikalytics(name)
            if data and isinstance(data, dict):
                if "tables" in data and data["tables"]:
                    for table in data["tables"]:
                        cols = [str(c).lower() for c in table.columns]
                        mv_col = next((c for c in cols if "move" in c), None)
                        if mv_col:
                            orig_col = [c for c in table.columns if str(c).lower() == mv_col][0]
                            pikalytics_moves = [str(v) for v in table[orig_col].dropna().tolist()]
                            if pikalytics_moves:
                                break
        except Exception:
            pikalytics_moves = []

    # Build ordered list: pikalytics-preferred then remaining
    move_by_name = {str(m.name): m for m in candidate_moves}
    ordered: List[Move] = []
    for mv_name in pikalytics_moves:
        if mv_name in move_by_name and mv_name not in [m.name for m in ordered]:
            ordered.append(move_by_name[mv_name])
    for m in candidate_moves:
        if m.name not in [x.name for x in ordered]:
            ordered.append(m)

    # Heuristic selection to fill 4 moves
    final: List[Move] = []
    stab_moves = [m for m in ordered if any(m.type == t for t in pkm.type)]
    if stab_moves:
        final.append(stab_moves[0])
    if len(final) < 2 and len(stab_moves) > 1:
        final.append(stab_moves[1])

    coverage_pool = [m for m in ordered if m.name not in [mv.name for mv in final] and (m.type not in pkm.type)]
    if coverage_pool:
        final.append(coverage_pool[0])

    idx = 0
    while len(final) < MAX_MOVES_PER_MON and idx < len(ordered):
        mv = ordered[idx]
        if mv.name not in [m.name for m in final]:
            final.append(mv)
        idx += 1

    while len(final) < MAX_MOVES_PER_MON:
        final.append(Move("Tackle", 40, "Normal", 100, 35, "physical"))

    # Prefer a utility/status move as 4th if we have STAB + coverage
    stab_count = sum(1 for m in final if any(m.type == t for t in pkm.type))
    if stab_count >= 1:
        for m in ordered:
            ml = str(m.name).lower()
            if m.name not in [x.name for x in final] and any(kw in ml for kw in UTILITY_KEYWORDS):
                weakest_idx = None
                weakest_power = float("inf")
                for i, fm in enumerate(final):
                    if not any(fm.type == t for t in pkm.type):
                        if fm.power < weakest_power:
                            weakest_power = fm.power
                            weakest_idx = i
                if weakest_idx is None:
                    weakest_idx = len(final) - 1
                final[weakest_idx] = m
                break

    # Ensure unique and exactly MAX_MOVES_PER_MON
    unique: List[Move] = []
    for m in final:
        if m.name not in [u.name for u in unique]:
            unique.append(m)
        if len(unique) >= MAX_MOVES_PER_MON:
            break
    return unique[:MAX_MOVES_PER_MON]


 
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


 
def generate_random_team(
    stats_df,
    moves_df,
    abilities_df,
    n: int = 6,
    use_pikalytics: bool = False,
    item_style: str = "balanced",
    allowed_names: Optional[Set[str]] = None,
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
            p.moves = _choose_moves_for_pokemon_advanced(p, moves_df, abilities_df, use_pikalytics)
            team.append(p)
        except Exception:
            continue
    return team


def generate_balanced_team(
    stats_df,
    moves_df,
    abilities_df,
    n: int = 6,
    candidate_pool_size: int = 60,
    use_pikalytics: bool = False,
    item_style: str = "balanced",
    allowed_names: Optional[Set[str]] = None,
) -> Team:
    all_names = list(stats_df["pokemon"].unique())
    if allowed_names is not None:
        all_names = [nm for nm in all_names if nm in allowed_names]
    random.shuffle(all_names)
    pool_names = all_names[: min(candidate_pool_size, len(all_names))]

    candidates: List[Pokemon] = []
    for name in pool_names:
        try:
            p = create_pokemon_from_name(name, stats_df, moves_df, abilities_df, preferred_item=pick_item(item_style))
            p.moves = _choose_moves_for_pokemon_advanced(p, moves_df, abilities_df, use_pikalytics)
            candidates.append(p)
        except Exception:
            continue

    # If no candidates created, fallback to random
    if not candidates:
        return generate_random_team(stats_df, moves_df, abilities_df, n=n, use_pikalytics=use_pikalytics, item_style=item_style)

    # Greedy selection maximizing synergy
    team: Team = []
    remaining = candidates[:]
    remaining.sort(key=lambda x: (x.max_hp + getattr(x, "defense", 0) + getattr(x, "special_defense", 0)), reverse=True)

    while len(team) < n and remaining:
        best_candidate = None
        best_score = float("-inf")
        for cand in list(remaining):
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
            score *= (1.0 + (random.random() - 0.5) * 0.05)
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
        if len(p.moves) != MAX_MOVES_PER_MON:
            p.moves = _choose_moves_for_pokemon_advanced(p, moves_df, abilities_df, use_pikalytics)

    return team


def build_team_from_prompt(
    prompt: str, stats_df, moves_df, abilities_df, n: int = 6, use_pikalytics: bool = False, item_style: str = "balanced"
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
    "build_team_from_prompt",
    "pick_item",
]
