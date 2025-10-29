# data_loader.py
"""
Data loader and domain models.

- Loads local CSV files:
    gen9_pokemon_stats.csv
    gen9_pokemon_moves.csv
    gen9_pokemon_abilities.csv

- Provides:
    - Pokemon and Move classes used across the system
    - load_local_data(stats_path, moves_path, abilities_path)
    - create_pokemon_from_name(name, stats_df, moves_df, abilities_df)
    - optional fetch_pikalytics(pokemon_name) for enrichment (cached)
"""

import os
import pandas as pd
import requests
import time
from typing import List, Optional

CACHE_DIR = "pikalytics_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class Move:
    def __init__(self, name: str, power: int = 0, mtype: str = "Normal",
                 accuracy: int = 100, pp: int = 10, category: str = "physical",
                 is_special: bool = False):
        self.name = name
        self.power = power or 0
        self.type = mtype
        self.accuracy = int(accuracy) if accuracy is not None else 100
        self.pp = int(pp) if pp is not None else 10
        self.max_pp = self.pp
        self.category = category  # "physical" / "special" / "status"
        self.is_special = is_special
        # optional: boosts dict or flags may be added externally
        self.boosts = getattr(self, "boosts", None)

    def __repr__(self):
        return f"Move({self.name}, {self.type}, pow={self.power}, pp={self.pp})"


class Pokemon:
    def __init__(self, name: str, types: List[str], hp: int, attack: int,
                 defense: int, spattack: int, spdefense: int, speed: int,
                 moves: List[Move], ability: Optional[str] = None, item: Optional[str] = None,
                 level: int = 50):
        self.name = name
        self.type = types if isinstance(types, list) else [types]
        self.max_hp = int(hp)
        self.hp = int(hp)
        self.attack = int(attack)
        self.defense = int(defense)
        self.spattack = int(spattack)
        self.spdefense = int(spdefense)
        self.speed = int(speed)
        self.moves = moves[:]  # shallow copy
        self.ability = ability
        self.item = item
        self.level = level
        # Stat stages and status for full battle integration
        self.stat_stages = {"attack": 0, "defense": 0, "special_attack": 0, "special_defense": 0, "speed": 0,
                            "accuracy": 0, "evasion": 0}
        self.status = None
        # convenience
    def is_fainted(self):
        return self.hp <= 0

    def __repr__(self):
        return f"Pokemon({self.name}, types={self.type}, hp={self.hp}/{self.max_hp})"


# ---------------- Data Loading ----------------
def load_local_data(stats_path="gen9_pokemon_stats.csv",
                    moves_path="gen9_pokemon_moves.csv",
                    abilities_path="gen9_pokemon_abilities.csv"):
    """Return (stats_df, moves_df, abilities_df). Expects common columns: 'pokemon' lowercase or case-insensitive."""
    stats_df = pd.read_csv(stats_path)
    moves_df = pd.read_csv(moves_path)
    abilities_df = pd.read_csv(abilities_path)
    # unify column names to lowercase for convenience
    stats_df.columns = [c.lower() for c in stats_df.columns]
    moves_df.columns = [c.lower() for c in moves_df.columns]
    abilities_df.columns = [c.lower() for c in abilities_df.columns]
    return stats_df, moves_df, abilities_df


# ---------------- Helpers to build objects ----------------
def _choose_moves_for_pokemon(pokemon_name: str, moves_df: pd.DataFrame, max_moves=4) -> List[Move]:
    # moves_df expected to have columns: pokemon, move, type, power, accuracy, pp, category
    subset = moves_df[moves_df['pokemon'].str.lower() == pokemon_name.lower()]
    if subset.empty:
        return []  # no moves known
    # prefer moves with power and then status coverage — simple heuristic
    subset = subset.copy()
    subset['power'] = pd.to_numeric(subset.get('power', 0), errors='coerce').fillna(0)
    subset = subset.sort_values(by=['power', 'accuracy'], ascending=[False, False])
    chosen = subset.head(max_moves)
    move_objs = []
    for _, row in chosen.iterrows():
        name = row.get('move') or row.get('name')
        mtype = row.get('type') or "Normal"
        power = int(row.get('power') or 0)
        accuracy = int(row.get('accuracy') or 100)
        pp = int(row.get('pp') or 10)
        cat = row.get('category') or ("special" if row.get('is_special', False) else "physical")
        is_special = (cat == "special")
        move_objs.append(Move(name=name, power=power, mtype=mtype, accuracy=accuracy, pp=pp, category=cat, is_special=is_special))
    return move_objs


def create_pokemon_from_name(name: str, stats_df: pd.DataFrame, moves_df: pd.DataFrame, abilities_df: pd.DataFrame,
                             level: int = 50, preferred_item: Optional[str] = None):
    """Creates a Pokemon object based on CSV rows. Raises ValueError if not found."""
    row = stats_df[stats_df['pokemon'].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Pokemon '{name}' not found in stats CSV.")
    row = row.iloc[0]
    # columns that might appear: pokemon, type1, type2, hp, attack, defense, sp_attack/spattack, sp_defense/spdefense, speed
    t1 = row.get('type1') or row.get('type')
    t2 = row.get('type2') if 'type2' in row and not pd.isna(row.get('type2')) else None
    types = [t1] if pd.notna(t1) else []
    if t2 and pd.notna(t2):
        types.append(t2)

    hp = row.get('hp') or row.get('base_hp') or 50
    atk = row.get('attack') or row.get('atk') or 50
    defense = row.get('defense') or row.get('def') or 50
    spatk = row.get('sp_attack') or row.get('spatk') or row.get('sp_atk') or 50
    spdef = row.get('sp_defense') or row.get('spdef') or row.get('sp_def') or 50
    speed = row.get('speed') or 50

    moves = _choose_moves_for_pokemon(name, moves_df, max_moves=4)
    ability_row = abilities_df[abilities_df['pokemon'].str.lower() == name.lower()]
    ability = None
    if not ability_row.empty:
        # take the first ability column found
        ability = ability_row.iloc[0].get('ability') or ability_row.iloc[0].get('abilities')

    item = preferred_item
    return Pokemon(name=name, types=types, hp=hp, attack=atk, defense=defense,
                   spattack=spatk, spdefense=spdef, speed=speed, moves=moves, ability=ability, item=item, level=level)


# ---------------- Optional: Pikalytics fetcher (simple) ----------------
def fetch_pikalytics(pokemon_slug: str, use_cache=True, wait=1.0):
    """
    Fetch basic data from pikalytics page for a pokemon.
    pokemon_slug: e.g. 'Torkoal' or 'torkoal' or 'gen9ou/Torkoal' (optional path).
    This function uses pandas.read_html to find tables and caches the page html.
    Returns a dict with scraped tables (if any) or raises on network error.
    NOTE: This is a helper; Pikalytics layout can change. Use responsibly.
    """
    base = "https://www.pikalytics.com/pokedex"
    if "/" in pokemon_slug:
        url = f"{base}/{pokemon_slug}"
    else:
        url = f"{base}/gen9ou/{pokemon_slug}"

    cache_file = os.path.join(CACHE_DIR, f"{pokemon_slug.replace('/', '_')}.html")
    html = None
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            html = f.read()
    else:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        html = resp.text
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(html)
        time.sleep(wait)

    # parse tables with pandas
    try:
        tables = pd.read_html(html)
        return {"url": url, "tables": tables}
    except ValueError:
        # no tables found
        return {"url": url, "tables": []}
