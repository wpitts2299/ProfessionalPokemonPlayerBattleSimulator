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
from typing import List, Optional
import pandas as pd

CACHE_DIR = "pikalytics_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


 
class Move:
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
        self.category = category or ("special" if is_special else "physical")
        self.is_special = (self.category == "special")
        try:
            self.priority = int(priority) if priority is not None else 0
        except Exception:
            self.priority = 0
        self.effect = effect

    def __repr__(self):
        return f"Move({self.name}, {self.type}, pow={self.power}, pp={self.pp})"


class Pokemon:
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
        return p

    def __repr__(self):
        return f"Pokemon({self.name}, types={self.type}, hp={self.hp}/{self.max_hp}, ability={self.ability}, item={self.item})"


 
def load_local_data(stats_path="gen9_pokemon_stats.csv",
                    moves_path="gen9_pokemon_moves.csv",
                    abilities_path="gen9_pokemon_abilities.csv"):
    """
    Return (stats_df, moves_df, abilities_df).
    Normalizes column names to lowercase.
    """
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

    # ensure we have a 'pokemon' column for moves/abilities and either 'pokemon' or 'name' for stats
    # Normalize stats_df to use 'pokemon' column name if it contains 'pokemon' or 'name'
    if 'pokemon' not in stats_df.columns and 'name' in stats_df.columns:
        stats_df = stats_df.rename(columns={'name': 'pokemon'})
    if 'pokemon' not in moves_df.columns and 'name' in moves_df.columns:
        moves_df = moves_df.rename(columns={'name': 'pokemon'})
    if 'pokemon' not in abilities_df.columns and 'name' in abilities_df.columns:
        abilities_df = abilities_df.rename(columns={'name': 'pokemon'})

    # basic sanity
    for df, label in ((stats_df, "stats"), (moves_df, "moves"), (abilities_df, "abilities")):
        if 'pokemon' not in df.columns:
            raise ValueError(f"CSV for {label} must contain a 'pokemon' or 'name' column after normalization.")

    return stats_df, moves_df, abilities_df


 
def _choose_moves_for_pokemon(pokemon_name: str, moves_df: pd.DataFrame, max_moves=4) -> List[Move]:
    """
    Choose up to max_moves for a Pokémon using heuristics:
      - prefer STAB moves (same type)
      - ensure at least one coverage move if possible
      - include setup/status if available
      - fill to 4 moves
    """
    subset = moves_df[moves_df['pokemon'].str.lower() == pokemon_name.lower()]
    if subset.empty:
        # Provide a safe default if no moves are listed
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
        category = row.get('category') or ("special" if row.get('is_special', False) else "physical")
        effect = row.get('effect') if 'effect' in row.index else None
        mv = Move(mname, power, mtype, accuracy, pp, category, category == "special", effect=effect)

        # categorize
        if any(kw in str(mname).lower() for kw in setup_keywords):
            setup.append(mv)
        elif power > 60:
            # treat powerful moves as likely STAB/primary threats; we'll filter STAB later
            stab.append(mv)
        else:
            filler.append(mv)

    chosen = []

    # Add 1–2 strong STAB moves (later filtered by actual pokemon types in create_pokemon_from_name)
    for mv in sorted(stab, key=lambda m: m.power, reverse=True):
        if len(chosen) >= 2:
            break
        if mv.name not in [c.name for c in chosen]:
            chosen.append(mv)

    # Add a filler/coverage move if available
    if len(chosen) < 3:
        for mv in filler:
            if mv.name not in [c.name for c in chosen]:
                chosen.append(mv)
                break

    # Add a setup move if it makes sense
    if setup and len(chosen) < max_moves:
        chosen.append(setup[0])

    # Fill to max_moves
    idx = 0
    while len(chosen) < max_moves and idx < len(filler):
        mv = filler[idx]
        if mv.name not in [c.name for c in chosen]:
            chosen.append(mv)
        idx += 1

    return chosen[:max_moves]


def create_pokemon_from_name(name: str, stats_df: pd.DataFrame, moves_df: pd.DataFrame, abilities_df: pd.DataFrame,
                             level: int = 50, preferred_item: Optional[str] = None):
    """Create a Pokemon instance from CSV rows (defensive about column names)."""
    # Allow stats_df to use 'pokemon' column; try various common names
    key_col = 'pokemon' if 'pokemon' in stats_df.columns else 'name'
    row = stats_df[stats_df[key_col].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Pokemon '{name}' not found in stats CSV.")
    row = row.iloc[0]

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
    spatk = get_stat('special_attack', 'sp_atk', 'spatk', 'spa', default=50)
    defense = get_stat('defense', 'def', 'base_def', default=50)
    spdef = get_stat('special_defense', 'sp_def', 'spdef', default=50)
    speed = get_stat('speed', 'spe', default=50)

    # choose moves with heuristics
    moves = _choose_moves_for_pokemon(name, moves_df, max_moves=4)
    if len(moves) < 4:
        # try again with looser heuristics
        extra = _choose_moves_for_pokemon(name, moves_df, max_moves=6)
        for mv in extra:
            if len(moves) >= 4:
                break
            if mv.name not in [m.name for m in moves]:
                moves.append(mv)

    # ability retrieval
    ability = None
    if 'pokemon' in abilities_df.columns:
        ability_row = abilities_df[abilities_df['pokemon'].str.lower() == name.lower()]
        if not ability_row.empty:
            # allowances for columns named 'ability' or 'abilities'
            ability = ability_row.iloc[0].get('ability') or ability_row.iloc[0].get('abilities')

    # item selection
    item = preferred_item
    return Pokemon(name, types, hp, atk, spatk, defense, spdef, speed, moves, ability, item, level)


 
def fetch_pikalytics(pokemon_slug: str, use_cache=True, wait=1.0):
    """
    Fetch basic data from pikalytics page for a Pokémon.
    Caches raw HTML to CACHE_DIR/<pokemon>.html
    Returns {'url': url, 'tables': [pandas tables...]}.
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
