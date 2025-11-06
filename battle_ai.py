"""
Battle AI module (complete)

- Full 18-type chart (attacking -> defending multipliers)
- Simulates simplified Pokémon battles between two teams
- Handles damage calculation, weather, abilities, and immunities
- Uses recursive prediction to choose best actions
- Dynamically reads setup and weather moves from moves_df
- Weather extension items modify duration automatically
- AI avoids redundant weather setup unless weather is expiring (<=1 turn)
"""

import random
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any, Iterable

from data_loader import (
    Move,
    Pokemon,
    load_local_data,
    HAZARD_MOVES,
    CLEAR_HAZARD_MOVES,
    SCREEN_MOVES,
    FIELD_MOVES,
    TERRAIN_MOVES,
    WEATHER_MOVES,
)

 
# Values follow standard Pokemon multipliers: 2.0 super, 0.5 not very, 0.0 immune.
TYPE_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0, "Fairy": 0.5},
    "Water":    {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0, "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0, "Flying": 2.0, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5, "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0, "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Rock": 2.0, "Dark": 2.0, "Steel": 2.0, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0.0, "Fairy": 0.5},
    "Poison":   {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
    "Ground":   {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0, "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying":   {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5, "Fairy": 0.5},
    "Rock":     {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5, "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost":    {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon":   {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0, "Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
    "Fairy":    {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0, "Dark": 2.0, "Steel": 0.5},
}


def _norm_type(t: str) -> str:
    return (t or "").strip().title()


def get_type_multiplier(move_type: str, defender_types: List[str]) -> float:
    """Return combined multiplier for move_type vs defender_types (dual-type aware)."""
    mt = _norm_type(move_type)
    mult = 1.0
    for dt in defender_types:
        mult *= TYPE_EFFECTIVENESS.get(mt, {}).get(_norm_type(dt), 1.0)
    return mult


 
class BattleState:
    """
    Lightweight battle state for simulation. Teams are lists of Pokemon objects.
    Pokemon instances should be 'copy_for_battle' clones (so simulation is isolated).
    """

    def __init__(self, ai_team: List[Pokemon], player_team: List[Pokemon]):
        """Clone the incoming rosters and attach bookkeeping for weather, terrain, and side effects."""
        self.ai_team = [p.copy_for_battle() for p in ai_team]
        self.player_team = [p.copy_for_battle() for p in player_team]
        self.weather = {"type": None, "turns": 0}  # {"type": "Rain"/"Sun"/"Sand"/"Snow", "turns": int}
        self.field = {
            "terrain": {"type": None, "turns": 0},
            "trick_room": 0,
            "gravity": 0,
            "magic_room": 0,
            "wonder_room": 0,
        }
        self.side_conditions = {
            "ai": self._default_side_state(),
            "player": self._default_side_state(),
        }
        self.turn_half = 0
        self.turn_count = 0
        self.last_events: List[str] = []
        self.last_event: str = ""

    @staticmethod
    def _default_side_state() -> Dict[str, Any]:
        """Return the baseline side-condition structure for hazards, screens, and buffs."""
        return {
            "hazards": {"stealth_rock": False, "spikes": 0, "toxic_spikes": 0, "sticky_web": False},
            "screens": {"reflect": 0, "light_screen": 0, "aurora_veil": 0},
            "safeguard": 0,
            "mist": 0,
            "tailwind": 0,
            "lucky_chant": 0,
        }

    def active_ai(self) -> Optional[Pokemon]:
        """Return the first living AI Pokemon; used as the active combatant."""
        return next((p for p in self.ai_team if not p.is_fainted()), None)

    def active_player(self) -> Optional[Pokemon]:
        """Return the first living player Pokemon; used as the human-controlled active slot."""
        return next((p for p in self.player_team if not p.is_fainted()), None)

    def is_terminal(self) -> bool:
        """A battle ends when one side has no healthy Pokemon remaining."""
        return all(p.is_fainted() for p in self.ai_team) or all(p.is_fainted() for p in self.player_team)

    def copy(self) -> "BattleState":
        """Deep-copy the state so recursive simulations cannot mutate shared objects."""
        return deepcopy(self)


 
class BattleAI:
    """Battle AI that chooses actions using recursive simulation."""

    def __init__(self, recursion_depth: int = 2):
        """Load local CSV data and record how many plies to explore during lookahead."""
        self.stats_df, self.moves_df, self.abilities_df = load_local_data()
        self.recursion_depth = recursion_depth

    
    def _side_of(self, state: BattleState, pokemon: Optional[Pokemon]) -> Optional[str]:
        """Return 'ai' or 'player' to indicate which roster a Pokemon belongs to."""
        if pokemon is None:
            return None
        if pokemon in state.ai_team:
            return "ai"
        if pokemon in state.player_team:
            return "player"
        return None

    def _opponent_side(self, side: Optional[str]) -> Optional[str]:
        """Translate a side label to its opposing side."""
        if side == "ai":
            return "player"
        if side == "player":
            return "ai"
        return None

    def _has_ability(self, pokemon: Optional[Pokemon], ability_name: str) -> bool:
        """Case-insensitive ability comparison with None safety."""
        if not pokemon or not getattr(pokemon, "ability", None):
            return False
        return str(pokemon.ability).strip().lower() == ability_name.strip().lower()

    def _has_item(self, pokemon: Optional[Pokemon], item_name: str) -> bool:
        """Case-insensitive held-item comparison with None safety."""
        if not pokemon or not getattr(pokemon, "item", None):
            return False
        return str(pokemon.item).strip().lower() == item_name.strip().lower()

    def _is_grounded(self, pokemon: Optional[Pokemon]) -> bool:
        """Return True if the Pokemon is affected by ground hazards/terrain."""
        if not pokemon:
            return False
        types = {_norm_type(t) for t in getattr(pokemon, "type", [])}
        if "Flying" in types:
            return False
        if self._has_ability(pokemon, "levitate"):
            return False
        if self._has_item(pokemon, "air balloon"):
            return False
        return True

    def _ensure_event_log(self, state: BattleState) -> None:
        """Ensure the simulation state exposes a mutable `last_events` list for narration."""
        if not hasattr(state, "last_events") or state.last_events is None:
            state.last_events = []

    def get_setup_moves(self) -> List[str]:
        """Detect setup/status moves (raise stats) from moves CSV (effect or name)."""
        setup_moves = set()
        if "effect" in self.moves_df.columns:
            for _, row in self.moves_df.iterrows():
                eff = str(row.get("effect") or "").lower()
                name = row.get("move") or row.get("name")
                if not name:
                    continue
                if any(term in eff and "raise" in eff for term in ["attack", "defense", "speed", "special"]):
                    setup_moves.add(name)
        else:
            # fallback: name-based detection
            keywords = [
                "dance",
                "plot",
                "calm mind",
                "shell smash",
                "bulk up",
                "iron defense",
                "work up",
                "swords dance",
                "agility",
            ]
            for _, row in self.moves_df.iterrows():
                name = str(row.get("move") or row.get("name") or "")
                if any(k in name.lower() for k in keywords):
                    setup_moves.add(row.get("move") or row.get("name"))
        return list(setup_moves)

    def get_weather_moves(self) -> Dict[str, str]:
        """Return mapping move_name -> weather type (Rain, Sun, Sand, Snow) detected from moves_df."""
        weather_moves: Dict[str, str] = {}
        for _, row in self.moves_df.iterrows():
            name = row.get("move") or row.get("name")
            if not name:
                continue
            lower_name = str(name).strip().lower()
            if lower_name in WEATHER_MOVES:
                weather_moves[name] = WEATHER_MOVES[lower_name]
                continue
            eff = str(row.get("effect") or row.get("short_descripton") or "").lower()
            if "rain" in eff or "heavy rain" in eff:
                weather_moves[name] = "Rain"
            elif "sun" in eff or "harsh sunlight" in eff:
                weather_moves[name] = "Sun"
            elif "sandstorm" in eff:
                weather_moves[name] = "Sand"
            elif "snow" in eff or "hail" in eff:
                weather_moves[name] = "Snow"
        return weather_moves

    
    def choose_ai_action(self, state: BattleState) -> Dict[str, Any]:
        """
        Implements the prescribed decision flow for AI move selection.
        Priority order:
            1. Use super-effective coverage when it cannot be safely countered.
            2. Boost offensive stats if safe to set up.
            3. Boost defensive stats if under pressure.
            4. Choose coverage that remains neutral against likely switches.
            5. Switch to a safer teammate when threatened.
            6. Fallback to strongest available attack.
        """
        if state.is_terminal():
            return {"type": "noop"}

        ai_pkm = state.active_ai()
        pl_pkm = state.active_player()
        if not ai_pkm or not pl_pkm:
            return {"type": "noop"}

        weather_moves = self.get_weather_moves()
        player_bench = self._alive_bench(state, "player", exclude=pl_pkm.name)

        # Step 1: Look for high-value super-effective moves and ensure the opponent cannot safely pivot.
        super_moves = self._candidate_super_effective_moves(ai_pkm, pl_pkm, state)
        if super_moves:
            for dmg, mv in super_moves:
                if not player_bench:
                    return {"type": "attack", "move": mv, "reason": "flow_super_finish"}
                bench_weak = any(get_type_multiplier(mv.type, bench.type) >= 2.0 for bench in player_bench)
                if bench_weak:
                    return {"type": "attack", "move": mv, "reason": "flow_super_pressure"}
                bench_resist = [
                    bench
                    for bench in player_bench
                    if get_type_multiplier(mv.type, bench.type) <= 0.5
                    or get_type_multiplier(mv.type, bench.type) == 0.0
                ]
                if not bench_resist:
                    return {"type": "attack", "move": mv, "reason": "flow_super_no_resist"}
                punishable = False
                for bench in bench_resist:
                    reply = self._best_opponent_move(bench, ai_pkm, state)
                    if not reply:
                        continue
                    if get_type_multiplier(reply.type, ai_pkm.type) < 2.0:
                        continue
                    if not self._outspeeds(ai_pkm, bench, state):
                        continue
                    dmg_vs_bench = self.calculate_damage(ai_pkm, bench, mv, state)
                    if dmg_vs_bench >= 0.6 * bench.hp:
                        punishable = True
                        break
                if punishable:
                    return {"type": "attack", "move": mv, "reason": "flow_super_punish_switch"}

        # Step 2: Consider offensive setup if we are relatively safe.
        atk_boost_info = self._find_stat_boost_move(ai_pkm, ["attack", "special_attack"])
        if atk_boost_info:
            boost_move, boost_map = atk_boost_info
            already_boosted = all(ai_pkm.stat_stages.get(stat, 0) >= 4 for stat in boost_map)
            if not already_boosted and self._should_use_attack_boost(state, ai_pkm, pl_pkm):
                return {"type": "setup_stat", "move": boost_move, "reason": "flow_attack_boost"}

        # Step 3: If threatened, look for a defensive boost option.
        def_boost_info = self._find_stat_boost_move(ai_pkm, ["defense", "special_defense"])
        if def_boost_info:
            def_boost_move, def_map = def_boost_info
            already_boosted = all(ai_pkm.stat_stages.get(stat, 0) >= 4 for stat in def_map)
            if not already_boosted and self._should_use_defense_boost(state, ai_pkm, pl_pkm):
                return {"type": "setup_stat", "move": def_boost_move, "reason": "flow_defense_boost"}

        # Step 4: Choose coverage that remains neutral or better against the player's bench.
        coverage_move = self._pick_move_minimax_vs_switch(state, ai_pkm, pl_pkm, player_bench)
        if coverage_move:
            return {"type": "attack", "move": coverage_move, "reason": "flow_cover_switch"}

        # Step 5: If a switch threatens to overwhelm us, find a safer teammate.
        best_reply_move = self._best_opponent_move(pl_pkm, ai_pkm, state)
        if best_reply_move:
            incoming_dmg = self.calculate_damage(pl_pkm, ai_pkm, best_reply_move, state)
            current_mult = get_type_multiplier(best_reply_move.type, ai_pkm.type)
            if incoming_dmg >= 0.6 * ai_pkm.hp or current_mult >= 2.0:
                defensive_switch = self._pick_defensive_switch(state, ai_pkm, pl_pkm)
                if defensive_switch:
                    switch_mult = get_type_multiplier(best_reply_move.type, defensive_switch.type)
                    if switch_mult < current_mult:
                        return {"type": "switch", "pokemon": defensive_switch, "reason": "flow_defensive_switch"}

        # Step 6: Fallback – pick the strongest available attack (super-effective first, then highest damage).
        best_attack: Optional[Tuple[float, Move]] = None
        for mv in ai_pkm.moves:
            if mv.pp <= 0 or mv.power <= 0:
                continue
            dmg = self.calculate_damage(ai_pkm, pl_pkm, mv, state)
            mult = get_type_multiplier(mv.type, pl_pkm.type)
            score = (2 if mult >= 2.0 else 1) * dmg
            if not best_attack or score > best_attack[0]:
                best_attack = (score, mv)
        if best_attack:
            return {"type": "attack", "move": best_attack[1], "reason": "flow_best_attack"}

        # No viable attacks (struggle scenario) – attempt weather/setup fallback or noop.
        for mv in ai_pkm.moves:
            if mv.name in weather_moves and mv.pp > 0:
                return {"type": "setup", "move": mv, "weather": weather_moves[mv.name], "reason": "flow_weather_fallback"}

        return {"type": "noop"}

    
    def recursive_evaluate(self, state: BattleState, depth: int, maximizing: bool) -> float:
        """
        Basic recursive evaluator:
        - maximizing=False means assume opponent will choose best response (minimax-like)
        - depth controls lookahead
        - evaluation metric: sum(ai_hp) - sum(player_hp)
        """
        if depth <= 0 or state.is_terminal():
            return self.evaluate_state(state)

        if maximizing:
            best = -float("inf")
            ai_action = self.choose_ai_action(state)
            st = self.simulate_action(state.copy(), state.active_ai(), state.active_player(), ai_action, self.get_weather_moves())
            val = self.recursive_evaluate(st, depth - 1, False)
            best = max(best, val)
            return best
        else:
            player = state.active_player()
            ai = state.active_ai()
            if not player or not ai:
                return self.evaluate_state(state)
            moves_sorted = sorted([m for m in player.moves if m.pp > 0], key=lambda m: self._simple_move_score(m, player, ai, state), reverse=True)
            candidates = moves_sorted[:2] if moves_sorted else []
            if not candidates:
                return self.evaluate_state(state)
            best_for_player = float("inf")
            for mv in candidates:
                action = {"type": "attack", "move": mv}
                st = self.simulate_action(state.copy(), attacker=player, defender=ai, action=action, weather_moves=self.get_weather_moves())
                val = self.recursive_evaluate(st, depth - 1, True)
                best_for_player = min(best_for_player, val)
            return best_for_player

    
    
    def _stage_mult(self, stage: int) -> float:
        """Convert a stat stage (-6..+6) into the standard battle multiplier."""
        return (2 + stage) / 2.0 if stage >= 0 else 2.0 / (2 - stage)

    def _effective_speed(self, p: Pokemon, state: BattleState, side: Optional[str]) -> float:
        """Calculate speed after factoring in stages, status ailments, items, weather, and Tailwind."""
        base = float(getattr(p, "speed", 0))
        base *= self._stage_mult(p.stat_stages.get("speed", 0))
        if getattr(p, "status", None) == "paralysis":
            base *= 0.5
        ability = str(getattr(p, "ability", "") or "").lower()
        weather_type = state.weather.get("type")
        if ability == "chlorophyll" and weather_type == "Sun":
            base *= 2.0
        elif ability == "swift swim" and weather_type == "Rain":
            base *= 2.0
        elif ability == "sand rush" and weather_type == "Sand":
            base *= 2.0
        elif ability == "slush rush" and weather_type in {"Snow", "Hail"}:
            base *= 2.0
        elif ability == "quick feet" and getattr(p, "status", None):
            base *= 1.5
        item = str(getattr(p, "item", "") or "").lower()
        if item == "choice scarf":
            base *= 1.5
        elif item in {"iron ball", "machobrace", "power anklet", "power band", "power belt", "power bracer", "power lens", "power weight"}:
            base *= 0.5
        if side and state.side_conditions.get(side, {}).get("tailwind", 0) > 0:
            base *= 2.0
        return max(base, 1.0)

    def _outspeeds(self, a: Pokemon, d: Pokemon, state: BattleState) -> bool:
        """Return True when `a` should act before `d`, respecting Trick Room inversions."""
        side_a = self._side_of(state, a)
        side_d = self._side_of(state, d)
        speed_a = self._effective_speed(a, state, side_a)
        speed_d = self._effective_speed(d, state, side_d)
        if state.field.get("trick_room", 0) and not self._has_ability(a, "stall"):
            return speed_a <= speed_d
        return speed_a >= speed_d

    def _best_opponent_move(self, defender: Pokemon, target: Pokemon, state: BattleState) -> Optional[Move]:
        """Heuristically pick the scariest move the opponent could use on the given target."""
        if not defender or not defender.moves:
            return None
        scored = []
        for m in defender.moves:
            if m.pp <= 0:
                continue
            # Score by expected damage approximation
            dmg = self.calculate_damage(defender, target, m, state)
            scored.append((dmg, m))
        return max(scored, key=lambda x: x[0])[1] if scored else None

    def _bench_candidates(self, state: BattleState, exclude_name: str) -> List[Pokemon]:
        """Return AI-controlled Pokemon that are healthy and not the excluded name."""
        return [p for p in state.ai_team if p.name != exclude_name and not p.is_fainted()]

    # --- Battle effect helpers -------------------------------------------------
    def _alive_team(self, team: List[Pokemon]) -> List[Pokemon]:
        """Strip out fainted Pokemon from an arbitrary roster."""
        return [p for p in team if not p.is_fainted()]

    def _alive_bench(self, state: BattleState, side: str, exclude: Optional[str] = None) -> List[Pokemon]:
        """Return viable switch candidates for the requested side, skipping exclusions and the active slot."""
        pool = state.ai_team if side == "ai" else state.player_team
        bench = []
        for p in pool:
            if p.is_fainted():
                continue
            if exclude and p.name == exclude:
                continue
            active = state.active_ai() if side == "ai" else state.active_player()
            if active and p.name == active.name:
                continue
            bench.append(p)
        return bench

    def _candidate_super_effective_moves(
        self,
        attacker: Pokemon,
        defender: Pokemon,
        state: BattleState,
    ) -> List[Tuple[float, Move]]:
        """Return super-effective attacking options sorted by estimated damage output."""
        candidates: List[Tuple[float, Move]] = []
        for mv in attacker.moves:
            if mv.pp <= 0 or mv.power <= 0:
                continue
            mult = get_type_multiplier(mv.type, defender.type)
            if mult >= 2.0:
                dmg = self.calculate_damage(attacker, defender, mv, state)
                candidates.append((dmg, mv))
        candidates.sort(key=lambda x: (x[0], x[1].power), reverse=True)
        return candidates

    def _move_effectiveness_vs_team(
        self,
        move: Move,
        targets: Iterable[Pokemon],
    ) -> Tuple[float, float]:
        """Evaluate the best and worst effectiveness multipliers of `move` against the provided targets."""
        best = -float("inf")
        worst = float("inf")
        for target in targets:
            mult = get_type_multiplier(move.type, target.type)
            best = max(best, mult)
            worst = min(worst, mult)
        if best == -float("inf"):
            best = 1.0
            worst = 1.0
        return best, worst

    def _find_stat_boost_move(self, pokemon: Pokemon, stats: Iterable[str]) -> Optional[Tuple[Move, Dict[str, int]]]:
        """Search the move pool for a self-targeting status move that boosts the desired stats."""
        desired = {s.lower() for s in stats}
        best_choice: Optional[Tuple[int, Dict[str, int], Move]] = None
        for mv in pokemon.moves:
            if mv.pp <= 0 or getattr(mv, "category", "").lower() != "status":
                continue
            meta = getattr(mv, "metadata", {}) or {}
            changes = meta.get("stat_changes", [])
            total_delta = 0
            applies = False
            boost_map: Dict[str, int] = {}
            for ch in changes:
                if ch.get("target") != "self":
                    continue
                stat = str(ch.get("stat", "")).lower()
                if stat in desired and ch.get("stages", 0) > 0:
                    total_delta += ch.get("stages", 0)
                    boost_map[stat] = boost_map.get(stat, 0) + ch.get("stages", 0)
                    applies = True
            if applies:
                key = total_delta
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, boost_map, mv)
        if best_choice:
            return best_choice[2], best_choice[1]
        return None

    def _should_use_attack_boost(self, state: BattleState, ai_pkm: Pokemon, player_pkm: Pokemon) -> bool:
        """Determine whether taking a turn to raise offensive stats is worth the risk."""
        best_move = self._best_opponent_move(player_pkm, ai_pkm, state)
        if not best_move:
            return True
        dmg = self.calculate_damage(player_pkm, ai_pkm, best_move, state)
        if dmg >= ai_pkm.hp:
            return False
        if dmg <= 0.6 * ai_pkm.hp:
            return True
        return False

    def _should_use_defense_boost(self, state: BattleState, ai_pkm: Pokemon, player_pkm: Pokemon) -> bool:
        """Gauge whether a defensive buff will meaningfully reduce incoming pressure."""
        best_move = self._best_opponent_move(player_pkm, ai_pkm, state)
        if not best_move:
            return False
        dmg = self.calculate_damage(player_pkm, ai_pkm, best_move, state)
        if dmg >= ai_pkm.hp:
            return False
        if dmg >= 0.4 * ai_pkm.hp and not self._outspeeds(ai_pkm, player_pkm, state):
            return True
        return False

    def _pick_move_minimax_vs_switch(
        self,
        state: BattleState,
        attacker: Pokemon,
        defender: Pokemon,
        defender_bench: List[Pokemon],
    ) -> Optional[Move]:
        """Evaluate attacking moves by considering the worst-case switch-in outcome."""
        if not defender_bench:
            return None
        candidates: List[Tuple[float, float, float, Move]] = []
        for mv in attacker.moves:
            if mv.pp <= 0 or mv.power <= 0:
                continue
            mult_now = get_type_multiplier(mv.type, defender.type)
            worst_mult = mult_now
            for bench_mon in defender_bench:
                mult = get_type_multiplier(mv.type, bench_mon.type)
                worst_mult = min(worst_mult, mult)
            dmg_current = self.calculate_damage(attacker, defender, mv, state)
            candidates.append((worst_mult, dmg_current, mv.power, mv))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        if candidates[0][0] <= 0:
            return None
        return candidates[0][3]

    def _pick_defensive_switch(
        self,
        state: BattleState,
        ai_pkm: Pokemon,
        player_pkm: Pokemon,
    ) -> Optional[Pokemon]:
        """Find a teammate that better resists the opponent's projected attack."""
        best_move = self._best_opponent_move(player_pkm, ai_pkm, state)
        if not best_move:
            return None
        bench = self._alive_bench(state, "ai", exclude=ai_pkm.name)
        if not bench:
            return None
        scored: List[Tuple[float, float, Pokemon]] = []
        for cand in bench:
            mult = get_type_multiplier(best_move.type, cand.type)
            bulk = cand.max_hp + getattr(cand, "defense", 0) + getattr(cand, "special_defense", 0)
            scored.append((mult, float(-bulk), cand))
        scored.sort(key=lambda x: (x[0], x[1]))
        for mult, _, cand in scored:
            if mult == 0.0:
                return cand
        for mult, _, cand in scored:
            if mult <= 0.5:
                return cand
        return scored[0][2] if scored else None

    def _apply_stat_changes(self, pokemon: Pokemon, changes: List[Dict[str, Any]], events: List[str], source_name: str) -> None:
        """Apply stage adjustments described in move metadata and log the resulting narration."""
        if not pokemon or not changes:
            return
        stat_map = {
            "attack": "attack",
            "defense": "defense",
            "special_attack": "special_attack",
            "special_defense": "special_defense",
            "speed": "speed",
            "accuracy": "accuracy",
            "evasion": "evasion",
        }
        for change in changes:
            if not change:
                continue
            chance = int(change.get("chance", 100) or 100)
            if chance < 100 and random.randint(1, 100) > chance:
                continue
            stat_key = stat_map.get(str(change.get("stat", "")).lower())
            if not stat_key or stat_key not in pokemon.stat_stages:
                continue
            delta = int(change.get("stages", 0))
            if delta == 0:
                continue
            current = pokemon.stat_stages.get(stat_key, 0)
            new_val = max(-6, min(6, current + delta))
            if new_val == current:
                continue
            pokemon.stat_stages[stat_key] = new_val
            verb = "rose" if delta > 0 else "fell"
            magnitude = abs(delta)
            stage_text = " sharply" if magnitude == 2 else " drastically" if magnitude >= 3 else ""
            events.append(f"{pokemon.name}'s {stat_key.replace('_', ' ').title()}{stage_text} {verb} due to {source_name}.")

    def _apply_status(
        self,
        state: BattleState,
        target: Pokemon,
        status: str,
        events: List[str],
        source_name: str,
        chance: int = 100,
        force: bool = False,
        source_pokemon: Optional[Pokemon] = None,
    ) -> bool:
        """Apply a status effect if legal, respecting immunities, terrain, Safeguard, and RNG chance."""
        status = str(status or "").lower()
        if not target or not status:
            return False
        if chance < 100 and random.randint(1, 100) > chance:
            return False

        # Volatile statuses
        if status == "flinch":
            target.volatiles["flinch"] = True
            events.append(f"{target.name} flinched!")
            return True
        if status == "confusion":
            if force or "confusion" not in target.volatiles:
                target.volatiles["confusion"] = random.randint(2, 5)
                events.append(f"{target.name} became confused!")
            return True

        # Primary status conditions
        existing = getattr(target, "status", None)
        if existing and status not in {"badly_poison"} and not force:
            return False

        side = self._side_of(state, target)
        safeguard_active = bool(side and state.side_conditions.get(side, {}).get("safeguard", 0) > 0)
        if safeguard_active and not force:
            events.append(f"{target.name} is protected by Safeguard!")
            return False

        terrain_type = state.field.get("terrain", {}).get("type")
        grounded = self._is_grounded(target)
        if terrain_type == "Misty" and grounded and status in {"burn", "poison", "badly_poison", "paralysis", "sleep"} and not force:
            events.append(f"The misty terrain shielded {target.name}!")
            return False
        if terrain_type == "Electric" and grounded and status == "sleep" and not force:
            events.append(f"The electric terrain kept {target.name} awake!")
            return False

        types = {_norm_type(t) for t in getattr(target, "type", [])}
        ability = str(getattr(target, "ability", "") or "").lower()

        if status == "burn":
            if "Fire" in types and not force:
                return False
            if ability in {"water veil", "water bubble"} and not force:
                return False
        elif status in {"poison", "badly_poison"}:
            if "Steel" in types or "Poison" in types:
                return False
            if ability in {"immunity"} and not force:
                return False
        elif status == "paralysis":
            if "Electric" in types and not force:
                return False
            if ability in {"limber"} and not force:
                return False
        elif status == "sleep":
            if ability in {"insomnia", "vital spirit"} and not force:
                return False
        elif status == "freeze":
            if "Ice" in types and not force:
                return False

        target.status = status if status != "badly_poison" else "badly_poison"
        if status == "sleep":
            target.status_duration = random.randint(2, 4)
            events.append(f"{target.name} fell asleep!")
        elif status == "freeze":
            target.status_duration = 0
            events.append(f"{target.name} was frozen solid!")
        else:
            target.status_duration = 0
            pretty = status.replace("_", " ")
            events.append(f"{target.name} was afflicted with {pretty}!")
        if status == "badly_poison":
            target.badly_poison_counter = 1
        if ability == "synchronize" and source_pokemon and status in {"burn", "poison", "badly_poison", "paralysis"}:
            self._apply_status(state, source_pokemon, "poison" if status == "badly_poison" else status, events, f"{target.name}'s Synchronize", chance=100, force=True)
        return True

    def _apply_hazard(
        self,
        state: BattleState,
        target_side: Optional[str],
        hazard_type: str,
        events: List[str],
        source_name: str,
        defender: Optional[Pokemon] = None,
        attacker: Optional[Pokemon] = None,
    ) -> None:
        """Update the hazard slots for a side, handling Magic Bounce reflections and stacking rules."""
        if target_side is None or hazard_type is None:
            return
        target_store = state.side_conditions[target_side]["hazards"]
        reflected = False
        if defender and self._has_ability(defender, "magic bounce"):
            target_side = self._side_of(state, attacker)
            target_store = state.side_conditions[target_side]["hazards"]
            reflected = True
            events.append(f"{defender.name}'s Magic Bounce reflected the {source_name}!")

        if hazard_type == "stealth_rock":
            if not target_store["stealth_rock"]:
                target_store["stealth_rock"] = True
                side_name = "AI" if target_side == "ai" else "player"
                if not reflected:
                    events.append(f"Pointed stones float around the opposing side ({side_name}).")
        elif hazard_type == "spikes":
            target_store["spikes"] = min(3, target_store["spikes"] + 1)
            events.append("Spikes were scattered on the opposing side.")
        elif hazard_type == "toxic_spikes":
            target_store["toxic_spikes"] = min(2, target_store["toxic_spikes"] + 1)
            events.append("Poison spikes were scattered on the ground.")
        elif hazard_type == "sticky_web":
            target_store["sticky_web"] = True
            events.append("Sticky web spread out on the ground.")

    def _clear_hazards(self, state: BattleState, side: Optional[str], events: List[str]) -> None:
        """Remove all entry hazards from the specified side and narrate the cleanup."""
        if not side:
            return
        state.side_conditions[side]["hazards"] = {
            "stealth_rock": False,
            "spikes": 0,
            "toxic_spikes": 0,
            "sticky_web": False,
        }
        events.append("All entry hazards on the side were cleared.")

    def _clear_screens(self, state: BattleState, side: Optional[str], events: List[str]) -> None:
        """Reset light screen/reflect/aurora veil timers."""
        if not side:
            return
        state.side_conditions[side]["screens"] = {"reflect": 0, "light_screen": 0, "aurora_veil": 0}
        events.append("Protective barriers disappeared!")

    def _on_switch_out(self, state: BattleState, pokemon: Pokemon, events: List[str]) -> None:
        """Perform bookkeeping when a Pokemon leaves the field (e.g., Tailwind timers, volatiles)."""
        if not pokemon:
            return
        if self._has_ability(pokemon, "natural cure") and getattr(pokemon, "status", None):
            pokemon.status = None
            pokemon.status_duration = 0
            pokemon.badly_poison_counter = 0
            events.append(f"{pokemon.name} was cured of its status!")
        if self._has_ability(pokemon, "regenerator"):
            heal = max(1, int(pokemon.max_hp / 3))
            pokemon.hp = min(pokemon.max_hp, pokemon.hp + heal)
        pokemon.volatiles.clear()
        pokemon.leech_seed = False

    def _apply_switch_in_effects(self, state: BattleState, pokemon: Pokemon, side: Optional[str], events: List[str]) -> None:
        """Trigger entry hazards, ability-based curing, and on-entry weather effects."""
        if not pokemon or not side:
            return
        hazards = state.side_conditions[side]["hazards"]
        if all([
            not hazards.get("stealth_rock"),
            hazards.get("spikes", 0) == 0,
            hazards.get("toxic_spikes", 0) == 0,
            not hazards.get("sticky_web"),
        ]):
            return
        if self._has_item(pokemon, "heavy-duty boots") or self._has_ability(pokemon, "magic guard"):
            events.append(f"{pokemon.name} avoided hazards thanks to its protection!")
            return

        # Stealth Rock
        if hazards.get("stealth_rock"):
            multiplier = get_type_multiplier("Rock", pokemon.type)
            damage = max(1, int(pokemon.max_hp * 0.125 * multiplier))
            pokemon.hp = max(0, pokemon.hp - damage)
            events.append(f"{pokemon.name} was hurt by the pointed stones (-{damage}).")

        # Spikes
        if hazards.get("spikes", 0) > 0 and self._is_grounded(pokemon):
            layers = hazards["spikes"]
            percent = {1: 0.125, 2: 0.167, 3: 0.25}.get(layers, 0.125)
            damage = max(1, int(pokemon.max_hp * percent))
            pokemon.hp = max(0, pokemon.hp - damage)
            events.append(f"{pokemon.name} was hurt by the spikes (-{damage}).")

        # Toxic Spikes
        if hazards.get("toxic_spikes", 0) and self._is_grounded(pokemon):
            toxins = hazards["toxic_spikes"]
            types = {_norm_type(t) for t in getattr(pokemon, "type", [])}
            if "Poison" in types:
                hazards["toxic_spikes"] = 0
                events.append(f"{pokemon.name} absorbed the toxic spikes!")
            elif "Steel" not in types:
                target_status = "badly_poison" if toxins >= 2 else "poison"
                self._apply_status(state, pokemon, target_status, events, "Toxic Spikes", force=True)

        # Sticky Web
        if hazards.get("sticky_web") and self._is_grounded(pokemon):
            pokemon.stat_stages["speed"] = max(-6, pokemon.stat_stages.get("speed", 0) - 1)
            events.append(f"{pokemon.name}'s Speed fell because of the sticky web!")

    def _apply_confusion_damage(self, state: BattleState, pokemon: Pokemon, events: List[str]) -> None:
        """Resolve the self-hit roll for confusion using a 40-power typeless attack."""
        temp_move = Move("Confusion Hit", 40, "Normal", 100, 1, "physical")
        dmg = max(1, self.calculate_damage(pokemon, pokemon, temp_move, state))
        pokemon.hp = max(0, pokemon.hp - dmg)
        events.append(f"{pokemon.name} hurt itself in its confusion (-{dmg}).")
        if pokemon.is_fainted():
            events.append(f"{pokemon.name} fainted!")

    def _handle_start_of_turn_conditions(self, state: BattleState, pokemon: Pokemon, events: List[str]) -> bool:
        """Process flinch, sleep, confusion, and other gatekeeping effects before a move executes."""
        if not pokemon or pokemon.is_fainted():
            return False
        if pokemon.volatiles.get("flinch"):
            events.append(f"{pokemon.name} flinched and couldn't move!")
            pokemon.volatiles.pop("flinch", None)
            return False

        if pokemon.status == "sleep":
            if pokemon.status_duration <= 0:
                pokemon.status = None
                events.append(f"{pokemon.name} woke up!")
            else:
                pokemon.status_duration -= 1
                if pokemon.status_duration <= 0:
                    pokemon.status = None
                    events.append(f"{pokemon.name} woke up!")
                else:
                    events.append(f"{pokemon.name} is fast asleep.")
                    return False

        if pokemon.status == "freeze":
            if random.random() < 0.2:
                pokemon.status = None
                events.append(f"{pokemon.name} thawed out!")
            else:
                events.append(f"{pokemon.name} is frozen solid!")
                return False

        if pokemon.status == "paralysis":
            if random.random() < 0.25:
                events.append(f"{pokemon.name} is fully paralyzed! It can't move!")
                return False

        if "confusion" in pokemon.volatiles:
            pokemon.volatiles["confusion"] -= 1
            if pokemon.volatiles["confusion"] <= 0:
                pokemon.volatiles.pop("confusion", None)
                events.append(f"{pokemon.name} snapped out of confusion!")
            else:
                if random.random() < (1.0 / 3.0):
                    self._apply_confusion_damage(state, pokemon, events)
                    return False

        return True

    def _apply_weather_move(self, state: BattleState, attacker: Pokemon, weather_type: Optional[str], move: Optional[Move], events: List[str]) -> None:
        """Set or extend weather while respecting item-based duration modifiers and redundancies."""
        if not weather_type:
            return
        base_turns = 5
        item = str(getattr(attacker, "item", "") or "")
        if weather_type == "Sun" and item == "Heat Rock":
            base_turns = 8
        elif weather_type == "Rain" and item == "Damp Rock":
            base_turns = 8
        elif weather_type == "Sand" and item == "Smooth Rock":
            base_turns = 8
        elif weather_type == "Snow" and item == "Icy Rock":
            base_turns = 8
        state.weather = {"type": weather_type, "turns": base_turns}
        events.append(f"The weather turned {weather_type.lower()}!")
        if isinstance(move, Move):
            move.pp = max(0, move.pp - 1)

    def _apply_terrain(self, state: BattleState, attacker: Pokemon, terrain_type: Optional[str], move: Optional[Move], events: List[str]) -> None:
        """Update terrain state, avoiding redundant refreshes that waste a turn."""
        if not terrain_type:
            return
        duration = 5
        if self._has_item(attacker, "terrain extender"):
            duration = 8
        state.field["terrain"] = {"type": terrain_type, "turns": duration}
        events.append(f"{terrain_type} Terrain enveloped the battlefield!")
        if isinstance(move, Move):
            move.pp = max(0, move.pp - 1)

    def _apply_screen(self, state: BattleState, side: Optional[str], screen_type: str, attacker: Pokemon, move: Optional[Move], events: List[str]) -> None:
        """Handle Reflect, Light Screen, and Aurora Veil timers per side."""
        if not side:
            return
        duration = 5
        if self._has_item(attacker, "light clay"):
            duration = 8
        screen_store = state.side_conditions[side]["screens"]
        screen_store[screen_type] = duration
        pretty = screen_type.replace("_", " ").title()
        events.append(f"{pretty} was set up!")
        if isinstance(move, Move):
            move.pp = max(0, move.pp - 1)

    def _apply_field_effect(self, state: BattleState, side: Optional[str], field_info: Dict[str, str], attacker: Pokemon, move: Optional[Move], events: List[str]) -> None:
        """Apply global or side-based field effects such as Tailwind, Trick Room, or Gravity."""
        eff_type = field_info.get("type")
        scope = field_info.get("scope")
        if not eff_type:
            return
        if scope == "side" and side:
            if eff_type == "tailwind":
                state.side_conditions[side]["tailwind"] = 4
                events.append("Tailwind began to blow behind the team!")
            elif eff_type == "safeguard":
                state.side_conditions[side]["safeguard"] = 5
                events.append("A veil of Safeguard surrounded the team!")
            elif eff_type == "mist":
                state.side_conditions[side]["mist"] = 5
                events.append("Mist shrouded the team!")
            elif eff_type == "lucky_chant":
                state.side_conditions[side]["lucky_chant"] = 5
                events.append("A Lucky Chant protects the team from critical hits!")
        elif scope == "global":
            if eff_type == "trick_room":
                state.field["trick_room"] = 5
                events.append("Space twisted around the battlefield!")
            elif eff_type == "gravity":
                state.field["gravity"] = 5
                events.append("Gravity intensified!")
            elif eff_type == "magic_room":
                state.field["magic_room"] = 5
                events.append("Magic Room distorted held item effects!")
            elif eff_type == "wonder_room":
                state.field["wonder_room"] = 5
                events.append("Wonder Room swapped defenses!")
            elif eff_type == "haze":
                for mon in state.ai_team + state.player_team:
                    mon.stat_stages = {k: 0 for k in mon.stat_stages}
                events.append("All stat changes were reset!")
        if isinstance(move, Move):
            move.pp = max(0, move.pp - 1)

    def _apply_secondary_effects(
        self,
        state: BattleState,
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        events: List[str],
    ) -> None:
        """Work through metadata-driven side effects (status, stat drops, hazards, healing, etc.)."""
        meta = getattr(move, "metadata", {}) or {}
        status_effects = meta.get("status", [])
        for effect in status_effects:
            status_name = effect.get("status")
            if not status_name:
                continue
            target_label = effect.get("target", "target")
            chance = int(effect.get("chance", 100) or 100)
            if target_label == "self":
                self._apply_status(state, attacker, status_name, events, move.name, chance=chance, source_pokemon=defender)
            else:
                self._apply_status(state, defender, status_name, events, move.name, chance=chance, source_pokemon=attacker)

        stat_changes = meta.get("stat_changes", [])
        if stat_changes:
            self_changes = [c for c in stat_changes if c.get("target") == "self"]
            opp_changes = [c for c in stat_changes if c.get("target") != "self"]
            if self_changes:
                self._apply_stat_changes(attacker, self_changes, events, move.name)
            if opp_changes:
                self._apply_stat_changes(defender, opp_changes, events, move.name)

        hazard_meta = meta.get("hazards")
        if hazard_meta:
            target_side = self._opponent_side(self._side_of(state, attacker))
            self._apply_hazard(state, target_side, hazard_meta.get("type"), events, move.name, defender=defender, attacker=attacker)

        if meta.get("clear_hazards"):
            target = meta["clear_hazards"]
            if target == "self":
                side = self._side_of(state, attacker)
                self._clear_hazards(state, side, events)
            elif target == "both":
                self._clear_hazards(state, "ai", events)
                self._clear_hazards(state, "player", events)
                self._clear_screens(state, "ai", events)
                self._clear_screens(state, "player", events)
            elif target == "swap":
                ai_state = state.side_conditions["ai"]
                player_state = state.side_conditions["player"]
                ai_state["hazards"], player_state["hazards"] = player_state["hazards"], ai_state["hazards"]
                ai_state["screens"], player_state["screens"] = player_state["screens"], ai_state["screens"]
                events.append("The teams swapped their field conditions!")

    def _tick_weather(self, state: BattleState) -> None:
        """Advance weather duration counters and clear them when they expire."""
        wt = state.weather.get("type")
        turns = state.weather.get("turns", 0)
        if wt and turns > 0:
            turns -= 1
            if turns <= 0:
                state.weather = {"type": None, "turns": 0}
            else:
                state.weather["turns"] = turns

    def _tick_field_conditions(self, state: BattleState) -> None:
        """Decrease durations for terrain, Trick Room, Gravity, and other timed effects."""
        terrain = state.field.get("terrain", {})
        if terrain.get("type") and terrain.get("turns", 0) > 0:
            terrain["turns"] -= 1
            if terrain["turns"] <= 0:
                state.field["terrain"] = {"type": None, "turns": 0}
        for key in ["trick_room", "gravity", "magic_room", "wonder_room"]:
            turns = state.field.get(key, 0)
            if turns > 0:
                state.field[key] = max(0, turns - 1)
        for side in ["ai", "player"]:
            sc = state.side_conditions[side]
            for screen_key in sc["screens"]:
                if sc["screens"][screen_key] > 0:
                    sc["screens"][screen_key] = max(0, sc["screens"][screen_key] - 1)
            for timed_key in ["tailwind", "safeguard", "mist", "lucky_chant"]:
                if sc.get(timed_key, 0) > 0:
                    sc[timed_key] = max(0, sc[timed_key] - 1)

    def _apply_end_of_turn_effects(self, state: BattleState, events: List[str]) -> None:
        """Handle burn/poison chip, Leftovers recovery, weather damage, and leech effects."""
        active_pairs = [("ai", state.active_ai()), ("player", state.active_player())]
        for side, mon in active_pairs:
            if not mon or mon.is_fainted():
                continue
            ability = str(getattr(mon, "ability", "") or "").lower()
            if mon.status == "burn" and ability != "magic guard":
                dmg = max(1, mon.max_hp // 16)
                mon.hp = max(0, mon.hp - dmg)
                events.append(f"{mon.name} is hurt by its burn (-{dmg}).")
            elif mon.status == "poison" and ability != "magic guard":
                dmg = max(1, mon.max_hp // 8)
                mon.hp = max(0, mon.hp - dmg)
                events.append(f"{mon.name} is hurt by poison (-{dmg}).")
            elif mon.status == "badly_poison" and ability != "magic guard":
                counter = max(1, getattr(mon, "badly_poison_counter", 1))
                dmg = max(1, int(mon.max_hp * counter / 16))
                mon.hp = max(0, mon.hp - dmg)
                mon.badly_poison_counter = counter + 1
                events.append(f"{mon.name} is suffering from toxic damage (-{dmg}).")

            # Weather residual damage
            weather_type = state.weather.get("type")
            types = {_norm_type(t) for t in getattr(mon, "type", [])}
            if weather_type == "Sand" and ability not in {"magic guard", "overcoat"} and not types.intersection({"Rock", "Ground", "Steel"}):
                dmg = max(1, mon.max_hp // 16)
                mon.hp = max(0, mon.hp - dmg)
                events.append(f"{mon.name} is buffeted by the sandstorm (-{dmg}).")
            elif weather_type in {"Snow", "Hail"} and ability not in {"magic guard", "overcoat", "ice body"} and "Ice" not in types:
                dmg = max(1, mon.max_hp // 16)
                mon.hp = max(0, mon.hp - dmg)
                events.append(f"{mon.name} is pelted by the snow (-{dmg}).")

            # Item-based healing/damage
            item = str(getattr(mon, "item", "") or "").lower()
            if item == "leftovers":
                heal = max(1, mon.max_hp // 16)
                mon.hp = min(mon.max_hp, mon.hp + heal)
                events.append(f"{mon.name} restored a little HP with its Leftovers (+{heal}).")
            elif item == "black sludge":
                if "Poison" in types:
                    heal = max(1, mon.max_hp // 16)
                    mon.hp = min(mon.max_hp, mon.hp + heal)
                    events.append(f"{mon.name} was healed by Black Sludge (+{heal}).")
                else:
                    dmg = max(1, mon.max_hp // 8)
                    mon.hp = max(0, mon.hp - dmg)
                    events.append(f"{mon.name} was hurt by Black Sludge (-{dmg}).")

            # Terrain healing
            terrain = state.field.get("terrain", {}).get("type")
            if terrain == "Grassy" and self._is_grounded(mon):
                heal = max(1, mon.max_hp // 16)
                mon.hp = min(mon.max_hp, mon.hp + heal)
                events.append(f"{mon.name} restored HP from the grassy terrain (+{heal}).")

            # Ability-based recovery/boosts
            if ability == "rain dish" and state.weather.get("type") == "Rain":
                heal = max(1, mon.max_hp // 16)
                mon.hp = min(mon.max_hp, mon.hp + heal)
                events.append(f"{mon.name} restored HP with Rain Dish (+{heal}).")
            if ability == "dry skin":
                if state.weather.get("type") == "Rain":
                    heal = max(1, mon.max_hp // 8)
                    mon.hp = min(mon.max_hp, mon.hp + heal)
                    events.append(f"{mon.name}'s Dry Skin restored HP (+{heal}).")
                elif state.weather.get("type") == "Sun":
                    dmg = max(1, mon.max_hp // 8)
                    mon.hp = max(0, mon.hp - dmg)
                    events.append(f"{mon.name}'s Dry Skin was hurt by the sunlight (-{dmg}).")
            if ability == "ice body" and state.weather.get("type") in {"Snow", "Hail"}:
                heal = max(1, mon.max_hp // 16)
                mon.hp = min(mon.max_hp, mon.hp + heal)
                events.append(f"{mon.name}'s Ice Body healed it (+{heal}).")
            if ability == "speed boost":
                mon.stat_stages["speed"] = min(6, mon.stat_stages.get("speed", 0) + 1)
                events.append(f"{mon.name}'s Speed rose thanks to Speed Boost!")

        # Leech Seed transfer
        ai_active = state.active_ai()
        player_active = state.active_player()
        if ai_active and ai_active.leech_seed and player_active and not player_active.is_fainted():
            ability = str(getattr(ai_active, "ability", "") or "").lower()
            if ability != "magic guard":
                dmg = max(1, ai_active.max_hp // 8)
                ai_active.hp = max(0, ai_active.hp - dmg)
                player_active.hp = min(player_active.max_hp, player_active.hp + dmg)
                events.append(f"{ai_active.name} lost HP from Leech Seed (-{dmg}).")
        if player_active and player_active.leech_seed and ai_active and not ai_active.is_fainted():
            ability = str(getattr(player_active, "ability", "") or "").lower()
            if ability != "magic guard":
                dmg = max(1, player_active.max_hp // 8)
                player_active.hp = max(0, player_active.hp - dmg)
                ai_active.hp = min(ai_active.max_hp, ai_active.hp + dmg)
                events.append(f"{player_active.name} lost HP from Leech Seed (-{dmg}).")

    def _advance_turn(self, state: BattleState, events: List[str]) -> None:
        """Update half-turn counters and tick per-turn effects once both sides have acted."""
        state.turn_half = (state.turn_half + 1) % 2
        if state.turn_half == 0:
            state.turn_count += 1
            self._tick_weather(state)
            self._tick_field_conditions(state)
            self._apply_end_of_turn_effects(state, events)

    def _rule_layer_action(self, state: BattleState, setup_moves: List[str], weather_moves: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """High-priority tactical checks (secure KOs, finish low-HP foes) before deeper evaluation."""
        ai_pkm = state.active_ai()
        pl_pkm = state.active_player()
        if not ai_pkm or not pl_pkm:
            return None

        # 1) If we have a super-effective (>=2x) attack that likely secures KO and we outspeed, use it
        best_lethal = None
        for mv in ai_pkm.moves:
            if mv.power <= 0 or mv.pp <= 0:
                continue
            mult = get_type_multiplier(mv.type, pl_pkm.type)
            if mult >= 2.0:  # 2x or 4x
                dmg = self.calculate_damage(ai_pkm, pl_pkm, mv, state)
                if dmg >= 0.6 * pl_pkm.hp and self._outspeeds(ai_pkm, pl_pkm, state):
                    # choose the strongest of these
                    if not best_lethal or dmg > best_lethal[0]:
                        best_lethal = (dmg, mv)
        if best_lethal is not None:
            return {"type": "attack", "move": best_lethal[1], "reason": "rule_super_effective_ko"}

        # 2) If opponent likely KOs us next turn, look for an immunity/resist switch; else best neutral tank
        opp_best = self._best_opponent_move(pl_pkm, ai_pkm, state)
        if opp_best is not None:
            opp_dmg = self.calculate_damage(pl_pkm, ai_pkm, opp_best, state)
            threatened = opp_dmg >= 0.7 * ai_pkm.hp  # threshold
            if threatened:
                bench = self._bench_candidates(state, ai_pkm.name)
                # Try immunity first (mult==0)
                immune = []
                resist = []
                neutral = []
                for cand in bench:
                    mult = get_type_multiplier(opp_best.type, cand.type)
                    if mult == 0.0:
                        immune.append(cand)
                    elif mult <= 0.5:
                        resist.append(cand)
                    else:
                        neutral.append((mult, cand))
                if immune:
                    return {"type": "switch", "pokemon": immune[0], "reason": "rule_switch_immunity"}
                if resist:
                    # pick the bulkiest resister
                    res = sorted(resist, key=lambda p: (p.max_hp + getattr(p, "defense", 0) + getattr(p, "special_defense", 0)), reverse=True)[0]
                    return {"type": "switch", "pokemon": res, "reason": "rule_switch_resist"}
                if neutral:
                    # choose best neutral tank (lowest multiplier, then bulk)
                    neutral.sort(key=lambda x: (x[0], -(x[1].max_hp + getattr(x[1], "defense", 0) + getattr(x[1], "special_defense", 0))))
                    return {"type": "switch", "pokemon": neutral[0][1], "reason": "rule_switch_neutral"}

        # 3) Defensive setup if threatened, cannot outspeed, and have a defense-boosting move
        if opp_best is not None:
            opp_dmg = self.calculate_damage(pl_pkm, ai_pkm, opp_best, state)
            if opp_dmg >= 0.5 * ai_pkm.hp and not self._outspeeds(ai_pkm, pl_pkm, state):
                for mv in ai_pkm.moves:
                    nm = str(mv.name).lower()
                    if any(k in nm for k in ["iron defense", "acid armor", "barrier", "amnesia", "cosmic power", "bulk up"]):
                        return {"type": "setup_stat", "move": mv, "reason": "rule_defensive_setup"}

        # 4) Weather: if about to expire (<=1) and we carry matching weather move, refresh
        if state.weather.get("turns", 0) <= 1:
            current = state.weather.get("type")
            for mv in ai_pkm.moves:
                wt = weather_moves.get(mv.name)
                if wt and (current != wt or state.weather.get("turns", 0) <= 1):
                    return {"type": "setup", "move": mv, "weather": wt, "reason": "rule_weather_refresh"}

        return None

    def simulate_action(self, state: BattleState, attacker: Pokemon, defender: Pokemon, action: Dict[str, Any], weather_moves: Dict[str, str]) -> BattleState:
        """
        Apply action (attack/setup/switch) to 'state' and return modified state.
        Attacker/defender are references to Pokemon inside state (use state's active_ai/active_player).
        """
        self._ensure_event_log(state)
        events: List[str] = []

        if attacker is None or attacker.is_fainted():
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        side = self._side_of(state, attacker)
        opp_side = self._opponent_side(side)
        action_type = action.get("type", "attack")
        move: Optional[Move] = action.get("move")

        if action_type == "switch":
            target_mon = action.get("pokemon")
            if not target_mon or target_mon.is_fainted():
                events.append("No healthy Pokémon to switch to!")
                self._advance_turn(state, events)
                state.last_events = events
                state.last_event = "\n".join(events)
                return state
            team = state.ai_team if side == "ai" else state.player_team
            for idx, mon in enumerate(team):
                if mon.name == target_mon.name and not mon.is_fainted():
                    team.insert(0, team.pop(idx))
                    break
            events.append(f"{attacker.name} was withdrawn!")
            self._on_switch_out(state, attacker, events)
            entrant = team[0]
            events.append(f"{entrant.name} entered the battle!")
            self._apply_switch_in_effects(state, entrant, side, events)
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if defender is None or defender.is_fainted():
            defender = state.active_player() if side == "ai" else state.active_ai()
        if defender and defender.is_fainted():
            defender = None

        if not self._handle_start_of_turn_conditions(state, attacker, events):
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if move is None:
            events.append("But nothing happened!")
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if move.pp <= 0:
            events.append(f"{attacker.name} tried to use {move.name}, but it has no PP left!")
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        move_meta = getattr(move, "metadata", {}) or {}
        move_category = getattr(move, "category", "").lower()
        is_status_move = move_category == "status" or move.power <= 0

        weather_type = action.get("weather") or move_meta.get("weather") or weather_moves.get(move.name)
        if weather_type and (is_status_move or action_type == "setup"):
            events.append(f"{attacker.name} used {move.name}!")
            self._apply_weather_move(state, attacker, weather_type, move, events)
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if is_status_move and move_meta.get("terrain"):
            events.append(f"{attacker.name} used {move.name}!")
            self._apply_terrain(state, attacker, move_meta.get("terrain"), move, events)
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if is_status_move and move_meta.get("field"):
            events.append(f"{attacker.name} used {move.name}!")
            for info in move_meta["field"]:
                self._apply_field_effect(state, side, info, attacker, move, events)
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        if is_status_move:
            events.append(f"{attacker.name} used {move.name}!")
            executed = False

            screen_type = move_meta.get("screen")
            if screen_type:
                self._apply_screen(state, side, screen_type, attacker, move, events)
                executed = True

            hazard_info = move_meta.get("hazards")
            if hazard_info:
                target_side = self._opponent_side(side)
                self._apply_hazard(state, target_side, hazard_info.get("type"), events, move.name, defender=defender, attacker=attacker)
                executed = True

            status_effects = move_meta.get("status", [])
            for effect in status_effects:
                status_name = effect.get("status")
                if not status_name:
                    continue
                target_label = effect.get("target", "target")
                chance = int(effect.get("chance", 100) or 100)
                if target_label == "self":
                    applied = self._apply_status(state, attacker, status_name, events, move.name, chance=chance, source_pokemon=defender)
                    executed = applied or executed
                else:
                    if defender and self._has_ability(defender, "magic bounce"):
                        events.append(f"{defender.name}'s Magic Bounce reflected the {move.name}!")
                        reflected = self._apply_status(state, attacker, status_name, events, move.name, chance=chance, force=True, source_pokemon=defender)
                        executed = reflected or executed
                    else:
                        applied = self._apply_status(state, defender, status_name, events, move.name, chance=chance, source_pokemon=attacker)
                        executed = applied or executed

            stat_changes = move_meta.get("stat_changes", [])
            if stat_changes:
                self_changes = [c for c in stat_changes if c.get("target") in {None, "self"}]
                target_changes = [c for c in stat_changes if c.get("target") == "target"]
                if self_changes:
                    self._apply_stat_changes(attacker, self_changes, events, move.name)
                    executed = True
                if target_changes and defender:
                    self._apply_stat_changes(defender, target_changes, events, move.name)
                    executed = True

            if move.name.lower() == "leech seed" and defender:
                defender.leech_seed = True
                events.append(f"{defender.name} was seeded!")
                executed = True

            if move_meta.get("healing"):
                heal_ratio = 0.5
                lower = move.effect_text.lower()
                if "one third" in lower or "1/3" in lower:
                    heal_ratio = 1 / 3
                heal = max(1, int(attacker.max_hp * heal_ratio))
                attacker.hp = min(attacker.max_hp, attacker.hp + heal)
                events.append(f"{attacker.name} restored HP (+{heal}).")
                executed = True

            if move_meta.get("cure_status"):
                if getattr(attacker, "status", None):
                    attacker.status = None
                    attacker.status_duration = 0
                    attacker.badly_poison_counter = 0
                    events.append(f"{attacker.name}'s status conditions were cured.")
                    executed = True
                attacker.volatiles.clear()

            clear_target = move_meta.get("clear_hazards")
            if clear_target:
                if clear_target == "self":
                    self._clear_hazards(state, side, events)
                elif clear_target == "both":
                    self._clear_hazards(state, "ai", events)
                    self._clear_hazards(state, "player", events)
                    self._clear_screens(state, "ai", events)
                    self._clear_screens(state, "player", events)
                elif clear_target == "swap":
                    ai_state = state.side_conditions["ai"]
                    pl_state = state.side_conditions["player"]
                    ai_state["hazards"], pl_state["hazards"] = pl_state["hazards"], ai_state["hazards"]
                    ai_state["screens"], pl_state["screens"] = pl_state["screens"], ai_state["screens"]
                    events.append("Court Change swapped the field conditions!")
                executed = True

            if not executed:
                events.append("But nothing happened!")

            attacker.last_move = move.name
            attacker.last_move_turn = 0
            move.pp = max(0, move.pp - 1)
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        events.append(f"{attacker.name} used {move.name}!")
        accuracy = move.accuracy or 100
        ability_att = str(getattr(attacker, "ability", "") or "").lower()
        ability_def = str(getattr(defender, "ability", "") or "").lower() if defender else ""

        if accuracy <= 0:
            hit_success = True
        else:
            acc_stage = attacker.stat_stages.get("accuracy", 0)
            eva_stage = defender.stat_stages.get("evasion", 0) if defender else 0
            acc_mult = self._stage_mult(acc_stage)
            eva_mult = self._stage_mult(eva_stage)
            acc_prob = (accuracy / 100.0) * (acc_mult / eva_mult)
            if ability_att == "compound eyes":
                acc_prob *= 1.3
            if ability_att == "hustle" and move.category == "physical":
                acc_prob *= 0.8
            if ability_def == "tangled feet" and defender and defender.status == "confusion":
                acc_prob *= 0.5
            acc_prob = max(0.0, min(1.0, acc_prob))
            hit_success = random.random() <= acc_prob

        if not hit_success:
            move.pp = max(0, move.pp - 1)
            events.append("But it missed!")
            self._advance_turn(state, events)
            state.last_events = events
            state.last_event = "\n".join(events)
            return state

        crit_rate = 1.0 / 24.0
        if self._has_item(attacker, "scope lens") or self._has_item(attacker, "razor claw"):
            crit_rate *= 2
        if ability_att == "super luck":
            crit_rate *= 2
        is_crit = random.random() < crit_rate

        damage = self.calculate_damage(attacker, defender, move, state, critical=is_crit) if defender else 0
        type_multiplier = get_type_multiplier(move.type, defender.type) if defender else 1.0

        move.pp = max(0, move.pp - 1)
        if defender:
            defender.hp = max(0, defender.hp - damage)

        if getattr(attacker, "last_move", None) == move.name:
            attacker.last_move_turn = getattr(attacker, "last_move_turn", 0) + 1
        else:
            attacker.last_move = move.name
            attacker.last_move_turn = 0

        recoil_text = ""
        if damage > 0 and self._has_item(attacker, "life orb"):
            recoil = max(1, int(attacker.max_hp * 0.1))
            attacker.hp = max(0, attacker.hp - recoil)
            recoil_text = f" {attacker.name} is hurt by its Life Orb! (-{recoil})"

        lower_effect = move.effect_text.lower()
        if damage > 0 and ("half the damage dealt" in lower_effect or "the user recovers hp equal to the damage dealt" in lower_effect or "drains the target's hp" in lower_effect or "absorbs half the damage" in lower_effect):
            heal = max(1, damage // 2)
            attacker.hp = min(attacker.max_hp, attacker.hp + heal)
            recoil_text += f" {attacker.name} regained HP from the attack (+{heal})."

        eff_txt = ""
        if type_multiplier == 0.0:
            eff_txt = " It had no effect."
        elif type_multiplier >= 2.0:
            eff_txt = " It's super effective!"
        elif type_multiplier <= 0.5:
            eff_txt = " It's not very effective."
        crit_txt = " Critical hit!" if is_crit else ""
        events.append(f"It dealt {damage} damage.{crit_txt}{eff_txt}{recoil_text}")

        if defender and not defender.is_fainted():
            self._apply_secondary_effects(state, attacker, defender, move, events)
        elif defender and defender.is_fainted():
            events.append(f"{defender.name} fainted!")

        self._advance_turn(state, events)
        state.last_events = events
        state.last_event = "\n".join(events)
        return state

    def calculate_damage(self, attacker: Pokemon, defender: Pokemon, move: Move, state: BattleState, critical: bool = False) -> int:
        """
        Enhanced damage approximation that mirrors key competitive mechanics:
        - Base power scaling from Attack/Sp. Atk versus Defense/Sp. Def with stat stages.
        - Critical hits bypassing negative offensive stages and positive defensive stages.
        - Same-type attack bonus (STAB), type matchups, weather, terrain, and ability/item boosts.
        - Screens, burn halving, and random +-15% variance.
        """
        if defender is None:
            return 0
        if move.power <= 0 or getattr(move, "category", "").lower() == "status":
            return 0

        move_type = _norm_type(move.type)

        # Immunities via item/ability
        if move_type == "Ground":
            if self._has_item(defender, "air balloon") or self._has_ability(defender, "levitate"):
                return 0

        attacker_side = self._side_of(state, attacker)
        defender_side = self._side_of(state, defender)

        weather = state.weather or {"type": None, "turns": 0}
        terrain = state.field.get("terrain", {"type": None, "turns": 0})

        ability_att = str(getattr(attacker, "ability", "") or "").lower()
        ability_def = str(getattr(defender, "ability", "") or "").lower()

        is_special = bool(getattr(move, "is_special", False) or getattr(move, "category", "").lower() == "special")
        atk_stat = attacker.special_attack if is_special else attacker.attack
        def_stat = defender.special_defense if is_special else defender.defense

        # Status-based modifiers
        if not is_special:
            if getattr(attacker, "status", None) == "burn" and ability_att != "guts":
                atk_stat *= 0.5
            if ability_att == "guts" and getattr(attacker, "status", None):
                atk_stat *= 1.5
        if ability_att == "hustle" and not is_special:
            atk_stat *= 1.5
        if ability_att == "solar power" and weather.get("type") == "Sun":
            atk_stat *= 1.5
        if ability_att == "slow start":
            if not is_special:
                atk_stat *= 0.5

        # Apply stat stages
        atk_stage = attacker.stat_stages.get("special_attack" if is_special else "attack", 0)
        def_stage = defender.stat_stages.get("special_defense" if is_special else "defense", 0)
        if critical and def_stage > 0:
            def_stage = 0
        atk_stat *= self._stage_mult(atk_stage)
        def_stat *= self._stage_mult(def_stage)

        # Defensive abilities
        if getattr(defender, "status", None) and ability_def == "marvel scale":
            def_stat *= 1.5
        if ability_def == "flower gift" and weather.get("type") == "Sun" and is_special:
            def_stat *= 1.5

        def_stat = max(def_stat, 1.0)

        # STAB
        stab = 1.0
        if any(_norm_type(move.type) == _norm_type(t) for t in getattr(attacker, "type", [])):
            stab = 1.5
            if ability_att == "adaptability":
                stab = 2.0

        # Type effectiveness
        type_mult = get_type_multiplier(move.type, defender.type)
        if ability_att == "tinted lens" and type_mult < 1.0:
            type_mult *= 2.0
        if ability_def in {"filter", "solid rock"} and type_mult > 1.0:
            type_mult *= 0.75
        if ability_def == "multiscale" and defender.hp == defender.max_hp:
            type_mult *= 0.5

        # Weather multipliers
        weather_mult = 1.0
        if weather.get("type") and not self._has_item(attacker, "utility umbrella"):
            wt = weather["type"]
            if wt == "Rain":
                if move_type == "Water":
                    weather_mult *= 1.5
                elif move_type == "Fire":
                    weather_mult *= 0.5
            elif wt == "Sun":
                if move_type == "Fire":
                    weather_mult *= 1.5
                elif move_type == "Water":
                    weather_mult *= 0.5
            elif wt == "Sand" and move_type == "Rock":
                weather_mult *= 1.3
            elif wt in {"Snow", "Hail"} and move_type == "Ice":
                weather_mult *= 1.3

        # Terrain multipliers
        terrain_mult = 1.0
        terrain_type = terrain.get("type")
        if terrain_type == "Electric" and self._is_grounded(attacker) and move_type == "Electric":
            terrain_mult *= 1.3
        elif terrain_type == "Grassy":
            if self._is_grounded(attacker) and move_type == "Grass":
                terrain_mult *= 1.3
            if self._is_grounded(defender) and move_type == "Ground":
                terrain_mult *= 0.5
        elif terrain_type == "Misty" and self._is_grounded(defender) and move_type == "Dragon":
            terrain_mult *= 0.5
        elif terrain_type == "Psychic" and self._is_grounded(attacker) and move_type == "Psychic":
            terrain_mult *= 1.3

        # Screens (Reflect/Light Screen/Aurora Veil)
        screen_mult = 1.0
        ignore_screens = ability_att == "infiltrator"
        if not ignore_screens and defender_side:
            screens = state.side_conditions.get(defender_side, {}).get("screens", {})
            if screens.get("aurora_veil", 0) > 0:
                screen_mult *= 0.5
            else:
                if not is_special and screens.get("reflect", 0) > 0:
                    screen_mult *= 0.5
                if is_special and screens.get("light_screen", 0) > 0:
                    screen_mult *= 0.5

        # Item multipliers
        item = str(getattr(attacker, "item", "") or "").lower()
        item_mult = 1.0
        if item == "life orb":
            item_mult *= 1.3
        elif item == "choice band" and not is_special:
            item_mult *= 1.5
        elif item == "choice specs" and is_special:
            item_mult *= 1.5
        elif item == "expert belt" and type_mult > 1.0:
            item_mult *= 1.2
        elif item == "muscle band" and not is_special:
            item_mult *= 1.1
        elif item == "wise glasses" and is_special:
            item_mult *= 1.1
        elif item == "metronome" and getattr(attacker, "last_move", "") == move.name:
            streak = min(4, max(0, getattr(attacker, "last_move_turn", 0)))
            item_mult *= 1.0 + 0.1 * streak

        # Defender items
        def_item = str(getattr(defender, "item", "") or "").lower()
        if def_item == "chilan berry" and move_type == "Normal" and type_mult > 1.0:
            type_mult *= 0.5
        if def_item == "shuca berry" and move_type == "Ground" and type_mult > 1.0:
            type_mult *= 0.5
        if def_item == "occa berry" and move_type == "Fire" and type_mult > 1.0:
            type_mult *= 0.5
        if def_item == "passen berry" and move_type == "Flying" and type_mult > 1.0:
            type_mult *= 0.5

        # Ability-based damage adjustments
        if ability_def == "thick fat" and move_type in {"Fire", "Ice"}:
            type_mult *= 0.5
        if ability_att == "sheer force" and getattr(move, "metadata", {}).get("status"):
            item_mult *= 1.3

        # Base damage formula
        level_factor = (2 * getattr(attacker, "level", 50)) / 5 + 2
        base = ((level_factor * move.power * (atk_stat / max(1.0, def_stat))) / 50.0) + 2.0
        variance = random.uniform(0.85, 1.0)
        dmg = base * stab * type_mult * weather_mult * terrain_mult * item_mult * screen_mult * variance
        dmg = max(0.0, dmg)
        return int(dmg)

    
    def evaluate_state(self, state: BattleState) -> float:
        """Simple heuristic: sum AI HP - sum Player HP (could be extended)."""
        ai_hp = sum(p.hp for p in state.ai_team)
        player_hp = sum(p.hp for p in state.player_team)
        return float(ai_hp - player_hp)

    def _simple_move_score(self, move: Move, user: Pokemon, target: Pokemon, state: BattleState) -> float:
        """Quick single-step heuristic for move ranking (used for player's candidate selection)."""
        if move.power <= 0:
            return 30.0 if user.hp > user.max_hp * 0.5 else 10.0
        mult = get_type_multiplier(move.type, target.type)
        stab = 1.5 if any(_norm_type(move.type) == _norm_type(t) for t in user.type) else 1.0
        return move.power * mult * stab * (move.accuracy or 100) / 100.0

    def _weather_item_synergy(self, pokemon: Pokemon, weather_type: str) -> bool:
        """Return True if the held item extends the provided weather type's duration."""
        item = getattr(pokemon, "item", "")
        if (item == "Damp Rock" and weather_type == "Rain") or \
           (item == "Heat Rock" and weather_type == "Sun") or \
           (item == "Smooth Rock" and weather_type == "Sand") or \
           (item == "Icy Rock" and weather_type == "Snow"):
            return True
        return False


 
if __name__ == "__main__":
    from team_builder import generate_balanced_team

    stats_df, moves_df, abilities_df = load_local_data()
    ai_team = generate_balanced_team(stats_df, moves_df, abilities_df, n=6)
    pl_team = generate_balanced_team(stats_df, moves_df, abilities_df, n=6)

    state = BattleState(ai_team, pl_team)
    ai = BattleAI(recursion_depth=2)
    action = ai.choose_ai_action(state)
    print("AI action chosen:", action)
