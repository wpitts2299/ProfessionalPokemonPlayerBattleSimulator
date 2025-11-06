"""
data_loader.py

- Loads local CSV files:
    gen9_pokemon_stats.csv
    gen9_pokemon_moves.csv
    gen9_pokemon_abilities.csv

- Domain models:
    - Move
    - Pokemon

- Helpers:
    - load_local_data(stats_path, moves_path, abilities_path)
    - create_pokemon_from_name(name, stats_df, moves_df, abilities_df)
    - _choose_moves_for_pokemon(...)
    - fetch_pikalytics(...) (optional cached HTML reader)
"""
import os
import time
import difflib
import re
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd

CACHE_DIR = "pikalytics_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Common keywords used to detect setup/utility moves in Pikalytics data.
SETUP_MOVE_KEYWORDS = [
    "swords dance",
    "calm mind",
    "nasty plot",
    "bulk up",
    "dragon dance",
    "shell smash",
    "quiver dance",
    "agility",
    "tailwind",
    "trick room",
    "iron defense",
    "acid armor",
    "rock polish",
    "curse",
    "coaching",
    "belly drum",
    "autotomize",
    "work up",
    "howl",
    "spicy extract",
    "chilly reception",
    "double team",
    "substitute",
]


STATUS_KEYWORDS = {
    "burn": "burn",
    "paralyz": "paralysis",
    "badly poison": "badly_poison",
    "toxic": "badly_poison",
    "poison": "poison",
    "sleep": "sleep",
    "freeze": "freeze",
    "confus": "confusion",
    "flinch": "flinch",
    "drows": "sleep",
    "restor": "heal",
}

STAT_NAME_ALIASES: Dict[str, List[str]] = {
    "attack": ["attack", "att."],
    "defense": ["defense", "def."],
    "special_attack": ["special attack", "sp. atk", "special atk", "sp atk"],
    "special_defense": ["special defense", "sp. def", "special def", "sp def"],
    "speed": ["speed"],
    "accuracy": ["accuracy"],
    "evasion": ["evasiveness", "evasion"],
}

HAZARD_MOVES: Dict[str, Dict[str, Any]] = {
    "stealth rock": {"type": "stealth_rock"},
    "spikes": {"type": "spikes"},
    "toxic spikes": {"type": "toxic_spikes"},
    "sticky web": {"type": "sticky_web"},
    "stone axe": {"type": "stealth_rock"},
    "ceaseless edge": {"type": "spikes"},
}

CLEAR_HAZARD_MOVES: Dict[str, str] = {
    "rapid spin": "self",
    "mortal spin": "self",
    "tidy up": "both",
    "defog": "both",
    "court change": "swap",
}

SCREEN_MOVES: Dict[str, str] = {
    "reflect": "reflect",
    "light screen": "light_screen",
    "aurora veil": "aurora_veil",
}

FIELD_MOVES: Dict[str, Dict[str, str]] = {
    "tailwind": {"type": "tailwind", "scope": "side"},
    "trick room": {"type": "trick_room", "scope": "global"},
    "magic room": {"type": "magic_room", "scope": "global"},
    "wonder room": {"type": "wonder_room", "scope": "global"},
    "gravity": {"type": "gravity", "scope": "global"},
    "safeguard": {"type": "safeguard", "scope": "side"},
    "mist": {"type": "mist", "scope": "side"},
    "lucky chant": {"type": "lucky_chant", "scope": "side"},
    "haze": {"type": "haze", "scope": "global"},
}

TERRAIN_MOVES: Dict[str, str] = {
    "electric terrain": "Electric",
    "grassy terrain": "Grassy",
    "misty terrain": "Misty",
    "psychic terrain": "Psychic",
}

WEATHER_MOVES: Dict[str, str] = {
    "rain dance": "Rain",
    "sunny day": "Sun",
    "sandstorm": "Sand",
    "snowscape": "Snow",
    "hail": "Snow",
    "chilly reception": "Snow",
}

def _word_to_int(word: str) -> Optional[int]:
    """Convert spelled-out numerals (one, two, etc.) used in free-form effect text."""
    mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    return mapping.get(word.strip().lower())


def _extract_percentage(text: str) -> Optional[int]:
    """Pull an integer percentage out of narrative move descriptions."""
    match = re.search(r"(\d+)\s*%", text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _infer_stage_delta(text: str) -> int:
    """Guess how many stat stages a move raises/lowers based on keywords such as 'sharply'."""
    if "drastically" in text or "severely" in text:
        return 3
    if "sharply" in text or "greatly" in text:
        return 2
    match = re.search(r"by\s+(one|two|three|four|\d+)\s+stage", text)
    if match:
        tok = match.group(1)
        if tok.isdigit():
            return int(tok)
        val = _word_to_int(tok)
        if val is not None:
            return val
    return 1


def _infer_effect_chance(text: str) -> int:
    """Best-effort guess of a secondary-effect activation chance."""
    pct = _extract_percentage(text)
    if pct is not None:
        return pct
    if "$effect_chance" in text:
        return 30
    if "always" in text or "guaranteed" in text or "will" in text or "inflicts" in text:
        return 100
    if "may" in text or "chance" in text:
        return 30
    return 100


def _parse_effect_metadata(name: str, effect_text: Optional[str]) -> Dict[str, Any]:
    """Turn verbose effect text into structured metadata used by the simulator."""
    text = (effect_text or "").strip()
    lower = text.lower()
    name_lower = name.lower()
    meta: Dict[str, Any] = {
        "status": [],
        "stat_changes": [],
        "hazards": None,
        "clear_hazards": None,
        "screen": None,
        "weather": None,
        "terrain": None,
        "field": [],
        "healing": None,
        "force_switch": False,
        "cure_status": False,
    }

    if name_lower in WEATHER_MOVES:
        meta["weather"] = WEATHER_MOVES[name_lower]
    if name_lower in TERRAIN_MOVES:
        meta["terrain"] = TERRAIN_MOVES[name_lower]
    if name_lower in HAZARD_MOVES:
        meta["hazards"] = HAZARD_MOVES[name_lower]
    if name_lower in CLEAR_HAZARD_MOVES and CLEAR_HAZARD_MOVES[name_lower]:
        meta["clear_hazards"] = CLEAR_HAZARD_MOVES[name_lower]
    if name_lower in SCREEN_MOVES:
        meta["screen"] = SCREEN_MOVES[name_lower]
    if name_lower in FIELD_MOVES:
        meta["field"].append(FIELD_MOVES[name_lower])

    if not lower:
        return meta

    # Status effects (e.g., burn, paralysis)
    for key, status in STATUS_KEYWORDS.items():
        if key in lower:
            if status == "heal":
                meta["healing"] = {"target": "self"}
                continue
            chance = _infer_effect_chance(lower)
            target = "target"
            if "user" in lower and "target" not in lower and status not in ("burn", "poison", "badly_poison"):
                target = "self"
            meta["status"].append({"status": status, "target": target, "chance": chance})

    # Stat changes from description text
    sentences = [s.strip() for s in re.split(r"[\\.!\n]", lower) if s.strip()]
    for sentence in sentences:
        for stat, aliases in STAT_NAME_ALIASES.items():
            if not any(alias in sentence for alias in aliases):
                continue
            target = "target"
            if any(tok in sentence for tok in ["user", "its own", "itself", "self", "ally"]):
                target = "self"
            if any(term in sentence for term in ["raise", "boost", "increase", "amplify"]):
                delta = _infer_stage_delta(sentence)
                meta["stat_changes"].append({
                    "target": target,
                    "stat": stat,
                    "stages": delta,
                    "chance": _infer_effect_chance(sentence),
                })
            if any(term in sentence for term in ["lower", "drop", "reduce", "decrease"]):
                delta = -_infer_stage_delta(sentence)
                meta["stat_changes"].append({
                    "target": target,
                    "stat": stat,
                    "stages": delta,
                    "chance": _infer_effect_chance(sentence),
                })
            break

    # Healing moves
    if "restore" in lower or "recovers" in lower or "heal" in lower:
        meta["healing"] = {"target": "self"}
    if "cures the user's status" in lower or "cures the user" in lower:
        meta["cure_status"] = True

    if "forces the target to switch" in lower or "switches the target out" in lower:
        meta["force_switch"] = True

    return meta
 
class Move:
    """Lightweight move representation that normalizes CSV fields into battle-ready attributes."""
    def __init__(self, name: str, power: Optional[int] = 0, mtype: str = "Normal",
                 accuracy: Optional[int] = 100, pp: Optional[int] = 10, category: str = "physical",
                 is_special: bool = False, priority: int = 0, effect: Optional[str] = None):
        self.name = str(name)
        # defensive coercion
        try:
            self.power = int(power) if power is not None and str(power).strip() != "" else 0
        except Exception:
            self.power = 0
        self.type = str(mtype) if mtype is not None else "Normal"
        try:
            self.accuracy = int(accuracy) if accuracy is not None and str(accuracy).strip() != "" else 100
        except Exception:
            self.accuracy = 100
        try:
            self.pp = int(pp) if pp is not None and str(pp).strip() != "" else 10
        except Exception:
            self.pp = 10
        self.max_pp = self.pp
        cat = str(category or ("special" if is_special else "physical")).strip().lower()
        if cat not in {"physical", "special", "status"}:
            cat = "special" if is_special else "physical"
        self.category = cat
        self.is_special = (self.category == "special")
        self.is_status = (self.category == "status")
        try:
            self.priority = int(priority) if priority is not None else 0
        except Exception:
            self.priority = 0
        self.effect_text = str(effect or "")
        self.effect = self.effect_text  # backward compatibility
        self.metadata = _parse_effect_metadata(self.name, self.effect_text)

    def __repr__(self):
        return f"Move({self.name}, {self.type}, cat={self.category}, pow={self.power}, pp={self.pp})"


class Pokemon:
    """Mutable Pokemon model used during simulation with stat stages, status flags, and move set."""
    def __init__(self, name: str, types: List[str], hp: int, attack: int,
                 special_attack: int, defense: int, special_defense: int, speed: int,
                 moves: List[Move], ability: Optional[str] = None, item: Optional[str] = None,
                 level: int = 50):
        self.name = str(name)
        self.type = types if isinstance(types, list) else [types]
        self.max_hp = int(hp)
        self.hp = int(hp)
        self.attack = int(attack)
        self.special_attack = int(special_attack)
        self.defense = int(defense)
        self.special_defense = int(special_defense)
        self.speed = int(speed)
        self.moves = moves[:]  # shallow copy
        self.ability = ability
        self.item = item
        self.level = level

        # stat stages
        self.stat_stages = {
            "attack": 0, "defense": 0, "special_attack": 0,
            "special_defense": 0, "speed": 0, "accuracy": 0, "evasion": 0
        }
        self.status = None
        self.status_duration = 0
        self.badly_poison_counter = 0
        self.volatiles: Dict[str, Any] = {}
        self.leech_seed = False
        self.last_move: Optional[str] = None
        self.last_move_turn: int = 0

    # compatibility aliases used by battle_ai
    @property
    def spattack(self):
        return self.special_attack

    @property
    def spdefense(self):
        return self.special_defense

    def is_fainted(self):
        return self.hp <= 0

    def copy_for_battle(self):
        """Lightweight clone for simulation (moves shallow-copied but as new Move objects)."""
        mcopy = [Move(m.name, m.power, m.type, m.accuracy, m.pp, m.category,
                      m.is_special, m.priority, m.effect) for m in self.moves]
        p = Pokemon(self.name, list(self.type), self.max_hp, self.attack, self.special_attack,
                    self.defense, self.special_defense, self.speed, mcopy,
                    self.ability, self.item, self.level)
        p.hp = self.hp
        p.stat_stages = dict(self.stat_stages)
        p.status = self.status
        p.status_duration = self.status_duration
        p.badly_poison_counter = self.badly_poison_counter
        p.volatiles = dict(self.volatiles)
        p.leech_seed = self.leech_seed
        p.last_move = self.last_move
        p.last_move_turn = self.last_move_turn
        return p

    def __repr__(self):
        return f"Pokemon({self.name}, types={self.type}, hp={self.hp}/{self.max_hp}, ability={self.ability}, item={self.item})"


 
def load_local_data(
    stats_path: str = "gen9_pokemon_stats.csv",
    moves_path: str = "gen9_pokemon_moves.csv",
    abilities_path: str = "gen9_pokemon_abilities.csv",
):
    """
    Return three DataFrames (stats, moves, abilities) with normalized column names.

    Path selection:
        1. Use the provided path when the file exists.
        2. Fall back to well-known alternates inside the repo.
    Column normalization:
        - Lowercase all columns and replace spaces with underscores.
        - Ensure stats/abilities expose a `pokemon` column, and moves expose a `move` column.
    """
    """
    # Prefer provided paths if they exist; else fall back to common alternates in this repo
    def pick_path(primary: str, fallbacks):
        if os.path.exists(primary):
            return primary
        for p in fallbacks:
            if os.path.exists(p):
                return p
        return primary  # let it raise if truly missing

    stats_path = pick_path(stats_path, ["pokemon.csv"])  # replacement dataset in this repo
    moves_path = pick_path(moves_path, ["pokemon_moves.csv"])  # global move DB
    abilities_path = pick_path(abilities_path, ["pokemon_abilities.csv"])  # replacement dataset

    stats_df = pd.read_csv(stats_path)
    moves_df = pd.read_csv(moves_path)
    abilities_df = pd.read_csv(abilities_path)

    # normalize column names to lowercase with spaces converted to _
    def norm_cols(df):
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df

    stats_df = norm_cols(stats_df)
    moves_df = norm_cols(moves_df)
    abilities_df = norm_cols(abilities_df)

    # ensure we have a 'pokemon' column for stats/abilities and a 'move' column for moves
    if 'pokemon' not in stats_df.columns and 'name' in stats_df.columns:
        stats_df = stats_df.rename(columns={'name': 'pokemon'})
    if 'pokemon' not in abilities_df.columns and 'name' in abilities_df.columns:
        abilities_df = abilities_df.rename(columns={'name': 'pokemon'})
    if 'move' not in moves_df.columns:
        if 'name' in moves_df.columns:
            moves_df = moves_df.rename(columns={'name': 'move'})
        elif 'move_name' in moves_df.columns:
            moves_df = moves_df.rename(columns={'move_name': 'move'})

    # basic sanity
    if 'pokemon' not in stats_df.columns:
        raise ValueError("Stats CSV must contain a 'pokemon' or 'name' column after normalization.")
    if 'pokemon' not in abilities_df.columns:
        # If abilities mapping is not provided, create an empty frame to keep downstream logic simple
        import pandas as _pd  # avoid shadowing
        abilities_df = _pd.DataFrame(columns=["pokemon", "ability", "abilities"])  

    return stats_df, moves_df, abilities_df


def _resolve_stats_row(name: str, stats_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str], Optional[str], Optional[float]]:
    """Resolve a Pokémon name to a stats row using exact or fuzzy matching."""
    if not isinstance(name, str):
        name = str(name)
    query = name.strip()
    if not query:
        return None, None, None, None
    query_lower = query.lower()

    candidate_cols = [col for col in ("pokemon", "name", "original_name") if col in stats_df.columns]
    if not candidate_cols:
        candidate_cols = list(stats_df.columns)

    # Exact match attempt
    for col in candidate_cols:
        series = stats_df[col].dropna().astype(str).str.strip()
        if series.empty:
            continue
        matches = series[series.str.lower() == query_lower]
        if not matches.empty:
            idx = matches.index[0]
            row = stats_df.loc[idx]
            resolved = str(row.get("pokemon") or matches.iloc[0])
            return row, resolved, col, 1.0

    # Fuzzy search fallback
    best_info: Optional[Tuple[pd.Series, str, str, float]] = None
    for col in candidate_cols:
        series = stats_df[col].dropna().astype(str).str.strip()
        if series.empty:
            continue
        lower_map = {val.lower(): val for val in series}
        candidates = list(lower_map.keys())
        match_list = difflib.get_close_matches(query_lower, candidates, n=1, cutoff=0.72)
        if not match_list:
            continue
        match_lower = match_list[0]
        score = difflib.SequenceMatcher(None, query_lower, match_lower).ratio()
        original_val = lower_map[match_lower]
        match_series = series[series.str.lower() == match_lower]
        if match_series.empty:
            continue
        idx = match_series.index[0]
        row = stats_df.loc[idx]
        resolved = str(row.get("pokemon") or original_val)
        info = (row, resolved, col, score)
        if best_info is None or score > best_info[3]:
            best_info = info

    if best_info is not None:
        return best_info

    return None, None, None, None


def _is_setup_move(move_name: str) -> bool:
    """Return True when the move name matches a known setup/support pattern."""
    ml = str(move_name or "").lower()
    return any(keyword in ml for keyword in SETUP_MOVE_KEYWORDS)


def _select_moves_with_setup(candidate_moves: List[Move], max_moves: int = 4) -> List[Move]:
    """Select a move set prioritizing power while ensuring at least one setup/utility option."""
    unique_moves: List[Move] = []
    seen_names: List[str] = []
    for mv in candidate_moves:
        if not isinstance(mv, Move):
            continue
        if mv.name in seen_names:
            continue
        unique_moves.append(mv)
        seen_names.append(mv.name)
        if len(unique_moves) >= max_moves:
            break

    if not unique_moves:
        return []

    if len(unique_moves) < max_moves:
        return unique_moves

    def is_status(move: Move) -> bool:
        return str(getattr(move, "category", "")).lower() == "status"

    chosen = unique_moves[:max_moves]
    if all(not is_status(mv) for mv in chosen):
        # Swap the lowest priority move for a setup move when possible so the AI
        # has at least one non-attacking option for stat boosts or support.
        for mv in candidate_moves:
            if mv.name in seen_names:
                continue
            if is_status(mv) and _is_setup_move(mv.name):
                chosen[-1] = mv
                break
    return chosen[:max_moves]


def _lookup_move_in_db(move_name: str, moves_df: pd.DataFrame) -> Optional[Move]:
    """Pull the normalized Move object for a given name, tolerating casing and missing data."""
    """Lookup a move by name in a global move DB and construct a Move."""
    if not move_name or moves_df is None or moves_df.empty:
        return None
    mdf = moves_df
    if 'move' not in mdf.columns and 'name' in mdf.columns:
        mdf = mdf.rename(columns={'name': 'move'})
    if 'move' not in mdf.columns:
        return None
    row = mdf[mdf['move'].astype(str).str.lower() == str(move_name).strip().lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    mtype = r.get('type') or r.get('move_type') or "Normal"
    def as_int(val, default):
        try:
            v = r.get(val)
            if v is None or str(v).strip() == "":
                return default
            return int(float(v))
        except Exception:
            return default
    power = as_int('power', 0)
    accuracy = as_int('accuracy', 100)
    pp = as_int('pp', 10)
    priority = as_int('priority', 0)
    category_raw = (r.get('damage_class') or r.get('category') or "physical")
    category = str(category_raw).strip().lower()
    effect = r.get('short_descripton') or r.get('short_description') or r.get('effect')
    cat = "special" if category == "special" else ("status" if category == "status" else "physical")
    return Move(name=str(move_name), power=power, mtype=str(mtype), accuracy=accuracy, pp=pp,
                category=cat, is_special=(cat == "special"), priority=priority, effect=effect)


def _choose_moves_for_pokemon(pokemon_name: str, moves_df: pd.DataFrame, max_moves=4) -> List[Move]:
    """Load the most frequently-used moves for `pokemon_name` with setup fallback heuristics."""
    """
    Choose up to max_moves for a Pokémon using heuristics:
      - prefer STAB moves (same type)
      - ensure at least one coverage move if possible
      - include setup/status if available
      - fill to 4 moves
    """
    # Use either a per-Pokémon move mapping or global DB + Pikalytics fallback
    # Prefer Pikalytics-tracked moves first, then enrich from CSV
    # Pikalytics detail pages provide the most reliable ordering, so try them first.
    try:
        from pikalytics_util import fetch_details  # lazy import
        det = fetch_details(pokemon_name)
    except Exception:
        det = None
    if isinstance(det, dict):
        names = [nm for nm, _ in det.get('moves', [])]
        pika_moves: List[Move] = []
        for nm in names:
            mv = _lookup_move_in_db(nm, moves_df)
            if mv:
                pika_moves.append(mv)
            if len(pika_moves) >= max_moves + 4:
                break
        selected = _select_moves_with_setup(pika_moves, max_moves=max_moves)
        if selected:
            return selected

    subset = None
    if 'pokemon' in moves_df.columns:
        subset = moves_df[moves_df['pokemon'].astype(str).str.lower() == pokemon_name.lower()]
    if subset is None or subset.empty:
        # Try Pikalytics for move names, then enrich from global DB
        move_names: List[str] = []
        try:
            from pikalytics_util import fetch_details  # lazy import
            det = fetch_details(pokemon_name)
            if isinstance(det, dict):
                move_names = [nm for nm, _ in det.get('moves', [])]
        except Exception:
            move_names = []
        picked: List[Move] = []
        for nm in move_names:
            mv = _lookup_move_in_db(nm, moves_df)
            if mv and mv.name not in [m.name for m in picked]:
                picked.append(mv)
                if len(picked) >= max_moves:
                    break
        if picked:
            return picked
        # Fallback: safe default if nothing else available
        return [Move(name="Tackle", power=40, mtype="Normal", accuracy=100, pp=35, category="physical", is_special=False)]

    subset = subset.copy()
    # defensive handling: if columns missing, create them with defaults
    if 'power' in subset.columns:
        subset['power'] = pd.to_numeric(subset['power'], errors='coerce').fillna(0)
    else:
        subset['power'] = 0
    if 'accuracy' in subset.columns:
        subset['accuracy'] = pd.to_numeric(subset['accuracy'], errors='coerce').fillna(100)
    else:
        subset['accuracy'] = 100

    # Extract unique setup moves dynamically
    setup_keywords = ["dance", "plot", "mind", "boost", "charge", "bulk", "cosmic", "curse", "calm", "shell"]
    setup_moves = [m for m in subset.get('move', subset.get('name')).unique() if any(kw in str(m).lower() for kw in setup_keywords)]

    # Heuristic grouping
    filler, stab, setup = [], [], []

    for _, row in subset.iterrows():
        mname = row.get('move') or row.get('name')
        mtype = row.get('type') or "Normal"
        try:
            power = int(row.get('power') or 0)
        except Exception:
            power = 0
        try:
            accuracy = int(row.get('accuracy') or 100)
        except Exception:
            accuracy = 100
        try:
            pp = int(row.get('pp') or 10)
        except Exception:
            pp = 10
        category = row.get('category') or row.get('damage_class') or ("special" if row.get('is_special', False) else "physical")
        category_str = str(category).strip().lower()
        if category_str not in {"physical", "special", "status"}:
            category_str = "special" if row.get('is_special', False) else "physical"
        effect = None
        if 'effect' in row.index:
            effect = row.get('effect')
        if not effect and 'short_descripton' in row.index:
            effect = row.get('short_descripton')
        mv = Move(mname, power, mtype, accuracy, pp, category_str, category_str == "special", effect=effect)

        # categorize
        if any(kw in str(mname).lower() for kw in setup_keywords):
            setup.append(mv)
        elif power > 60:
            # treat powerful moves as likely STAB/primary threats; we'll filter STAB later
            stab.append(mv)
        else:
            filler.append(mv)

    stab_sorted = sorted(stab, key=lambda m: m.power, reverse=True)
    ordered_candidates = stab_sorted + filler + setup
    return _select_moves_with_setup(ordered_candidates, max_moves=max_moves)


def create_pokemon_from_name(name: str, stats_df: pd.DataFrame, moves_df: pd.DataFrame, abilities_df: pd.DataFrame,
                             level: int = 50, preferred_item: Optional[str] = None):
    """
    Create a fully-initialized `Pokemon` object from the raw CSV data.

    Steps:
    1. Resolve the closest stats row (permitting fuzzy matches and form names).
    2. Normalize typing and base stats with tolerant fallbacks.
    3. Assemble a move set using usage heuristics (Pikalytics first, CSV fallback).
    4. Choose a plausible ability by consulting Pikalytics, ability CSV, then stats CSV.
    5. Default the held item to the provided style hint when available.
    """
    original_query = str(name)
    row, resolved_name, matched_col, match_score = _resolve_stats_row(original_query, stats_df)
    if row is None:
        raise ValueError(f"Pokemon '{original_query}' not found in stats CSV.")
    if resolved_name and resolved_name.lower() != original_query.lower():
        try:
            score_disp = f" ({match_score:.2f})" if match_score is not None else ""
        except Exception:
            score_disp = ""
        print(f"Matched '{original_query}' to '{resolved_name}'{score_disp} via {matched_col or 'fuzzy match'}.")
    canonical_name = resolved_name or original_query

    # tolerate different column names for types & stats
    t1 = row.get('type1') or row.get('type_1') or row.get('type') or row.get('type_1')
    t2 = None
    if 'type2' in row.index:
        t2 = row.get('type2')
    elif 'type_2' in row.index:
        t2 = row.get('type_2')

    types = []
    if pd.notna(t1):
        types.append(str(t1))
    if pd.notna(t2):
        types.append(str(t2))

    # stats tolerant getters
    def get_stat(*keys, default=50):
        for k in keys:
            if k in row.index and pd.notna(row.get(k)):
                try:
                    return int(row.get(k))
                except Exception:
                    try:
                        return int(float(row.get(k)))
                    except Exception:
                        return default
        return default

    hp = get_stat('hp', 'base_hp', 'hp_base', default=50)
    atk = get_stat('attack', 'atk', 'base_atk', default=50)
    spatk = get_stat('special_attack', 'sp_attack', 'sp_atk', 'spatk', 'spa', default=50)
    defense = get_stat('defense', 'def', 'base_def', default=50)
    spdef = get_stat('special_defense', 'sp_defense', 'sp_def', 'spdef', default=50)
    speed = get_stat('speed', 'spe', default=50)

    # choose moves with heuristics
    # Fetch moves using the original player input so we respect forms like "Ursaluna-Bloodmoon".
    moves = _choose_moves_for_pokemon(original_query, moves_df, max_moves=4)
    if len(moves) < 4 and canonical_name.lower() != original_query.lower():
        # If the direct lookup fails, retry with the resolved canonical name to
        # catch entries that only exist in the CSV under the base form.
        alt_moves = _choose_moves_for_pokemon(canonical_name, moves_df, max_moves=4)
        for mv in alt_moves:
            if mv.name not in [m.name for m in moves]:
                moves.append(mv)
            if len(moves) >= 4:
                break
    if len(moves) < 4:
        # try again with looser heuristics
        extra = _choose_moves_for_pokemon(original_query, moves_df, max_moves=6)
        for mv in extra:
            if len(moves) >= 4:
                break
            if mv.name not in [m.name for m in moves]:
                moves.append(mv)

    # ability retrieval with fallbacks: Pikalytics -> abilities CSV -> stats CSV
    ability = None
    # 1) Try Pikalytics usage top ability
    try:
        from pikalytics_util import fetch_details  # type: ignore
        det = fetch_details(original_query)
        if isinstance(det, dict):
            alst = det.get('abilities', []) or []
            if alst:
                ability = str(alst[0][0])
    except Exception:
        pass
    # 2) abilities_df row
    if not ability and 'pokemon' in abilities_df.columns:
        ab_row = abilities_df[abilities_df['pokemon'].astype(str).str.lower() == name.lower()]
        if not ab_row.empty:
            val = ab_row.iloc[0].get('ability') or ab_row.iloc[0].get('abilities')
            if val is not None:
                try:
                    import ast
                    parsed = val if isinstance(val, list) else ast.literal_eval(str(val))
                    if isinstance(parsed, list) and parsed:
                        ability = str(parsed[0])
                    else:
                        ability = str(val)
                except Exception:
                    ability = str(val)
    # 3) stats_df ability columns
    if not ability:
        if 'ability1' in row.index and pd.notna(row.get('ability1')) and str(row.get('ability1')).strip():
            ability = str(row.get('ability1'))
        elif 'ability' in row.index and pd.notna(row.get('ability')) and str(row.get('ability')).strip():
            ability = str(row.get('ability'))
        elif 'ability2' in row.index and pd.notna(row.get('ability2')):
            ability = str(row.get('ability2'))
        elif 'ability_hidden' in row.index and pd.notna(row.get('ability_hidden')):
            ability = str(row.get('ability_hidden'))

    # item selection
    item = preferred_item
    display_name = original_query if original_query else canonical_name
    return Pokemon(display_name, types, hp, atk, spatk, defense, spdef, speed, moves, ability, item, level)


 
def fetch_pikalytics(pokemon_slug: str, use_cache=True, wait=1.0):
    """
    Fetch basic data from the Pikalytics Pokédex for a Pokémon.

    Strategy:
    - Try to load the cached HTML from `pikalytics_cache/`.
    - If cache miss and `requests` is available, fetch from the live site and persist it.
    - Parse any HTML tables using `pandas.read_html`, returning them for downstream analysis.
    """
    # Lazy import to avoid hard dependency if requests isn't installed
    try:
        import requests  # type: ignore
    except Exception:
        return {"url": None, "tables": []}
    base = "https://www.pikalytics.com/pokedex"
    url = f"{base}/gen9ou/{pokemon_slug.lower()}"
    cache_file = os.path.join(CACHE_DIR, f"{pokemon_slug.replace('/', '_')}.html")

    html = None
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            html = f.read()
    else:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            html = resp.text
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html)
            # be polite
            time.sleep(wait)
        except Exception:
            html = None

    if not html:
        return {"url": url, "tables": []}

    try:
        tables = pd.read_html(html)
        return {"url": url, "tables": tables}
    except Exception:
        return {"url": url, "tables": []}
