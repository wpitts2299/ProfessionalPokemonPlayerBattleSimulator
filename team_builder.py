# team_builder.py
"""
Team builder module.

- Uses data_loader to create Pokemon objects and assemble teams.
- Provides:
    - generate_random_team(stats_df, moves_df, abilities_df)
    - generate_balanced_team(...)
    - build_team_from_prompt(prompt, ...)
"""

import random
from collections import Counter
from typing import List, Optional

from data_loader import load_local_data, create_pokemon_from_name, fetch_pikalytics, Pokemon


def _pick_unique_names(stats_df, n=6, excluded=None):
    all_names = list(stats_df['pokemon'].unique())
    if excluded:
        all_names = [n for n in all_names if n not in excluded]
    random.shuffle(all_names)
    return all_names[:n]


def generate_random_team(stats_df, moves_df, abilities_df, n=6):
    chosen = _pick_unique_names(stats_df, n=n)
    team = []
    for name in chosen:
        team.append(create_pokemon_from_name(name, stats_df, moves_df, abilities_df))
    return team


def generate_balanced_team(stats_df, moves_df, abilities_df, n=6):
    """
    Naive balanced builder:
    - tries to ensure type diversity,
    - attempts to include at least one setup/support and one bulky mon,
    - uses simple heuristics based on base stats.
    """
    candidates = list(stats_df['pokemon'].unique())
    random.shuffle(candidates)
    team = []
    seen_types = set()

    # heuristic: choose bulky first (HP + defenses), then sweepers
    stats_df['bulk'] = (stats_df.get('hp', 0).fillna(50) + stats_df.get('defense', 0).fillna(50) + stats_df.get('sp_defense', 0).fillna(50))
    bulky_sorted = stats_df.sort_values(by='bulk', ascending=False)

    # pick 2 bulky/trappers
    for _, row in bulky_sorted.head(12).iterrows():
        if len(team) >= 2:
            break
        name = row['pokemon']
        if name not in [p.name for p in team]:
            team.append(create_pokemon_from_name(name, stats_df, moves_df, abilities_df))

    # fill remaining trying to diversify types
    for name in candidates:
        if len(team) >= n:
            break
        if name in [p.name for p in team]:
            continue
        row = stats_df[stats_df['pokemon'] == name].iloc[0]
        t1 = row.get('type1') or row.get('type')
        if t1 in seen_types:
            continue
        try:
            pkm = create_pokemon_from_name(name, stats_df, moves_df, abilities_df)
            team.append(pkm)
            for tt in pkm.type:
                seen_types.add(tt)
        except Exception:
            continue

    # fallback: fill randomly
    if len(team) < n:
        remaining = [name for name in stats_df['pokemon'].unique() if name not in [p.name for p in team]]
        for name in random.sample(remaining, k=n-len(team)):
            team.append(create_pokemon_from_name(name, stats_df, moves_df, abilities_df))
    return team


def build_team_from_prompt(prompt: str, stats_df, moves_df, abilities_df, n=6, use_pikalytics=False):
    """
    Very simple NLP-ish parser: looks for Pokemon names in prompt and keywords (sun/rain).
    If use_pikalytics True, tries to fetch usage tables for detected Pokemon and bias moves/items.
    """
    prompt_lower = prompt.lower()
    detected = []
    # detect pokemon names present in stats_df
    names = list(stats_df['pokemon'].unique())
    for name in names:
        if name.lower() in prompt_lower:
            detected.append(name)
            if len(detected) >= n:
                break

    team = []
    # add detected pokemon first
    for name in detected:
        try:
            team.append(create_pokemon_from_name(name, stats_df, moves_df, abilities_df))
        except Exception:
            pass

    # Keywords
    strategy = None
    if 'sun' in prompt_lower or 'solardance' in prompt_lower:
        strategy = 'sun'
    elif 'rain' in prompt_lower:
        strategy = 'rain'
    elif 'sand' in prompt_lower:
        strategy = 'sand'
    elif 'snow' in prompt_lower:
        strategy = 'snow'
    # Fill remaining using balanced builder but bias toward strategy
    while len(team) < n:
        candidate = None
        if strategy:
            # choose mons of a type that benefits from strategy
            if strategy == 'rain':
                pool = stats_df[stats_df['type1'] == 'Water']['pokemon'].tolist()
            elif strategy == 'sun':
                pool = stats_df[stats_df['type1'] == 'Fire']['pokemon'].tolist()
            elif strategy == 'sand':
                pool = stats_df[stats_df['type1'].isin(['Rock', 'Ground', 'Steel'])]['pokemon'].tolist()
            elif strategy == 'snow':
                pool = stats_df[stats_df['type1'] == 'Ice']['pokemon'].tolist()
            else:
                pool = stats_df['pokemon'].tolist()
            if pool:
                candidate = random.choice(pool)
        if not candidate:
            candidate = random.choice(list(stats_df['pokemon'].unique()))
        if candidate not in [p.name for p in team]:
            team.append(create_pokemon_from_name(candidate, stats_df, moves_df, abilities_df))
    return team
