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
from typing import Optional, Tuple, List, Dict, Any

from data_loader import Move, Pokemon, load_local_data

 
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
        self.ai_team = [p.copy_for_battle() for p in ai_team]
        self.player_team = [p.copy_for_battle() for p in player_team]
        self.weather = {"type": None, "turns": 0}  # {"type": "Rain"/"Sun"/"Sand"/"Snow", "turns": int}
        self.last_event: str = ""

    def active_ai(self) -> Optional[Pokemon]:
        return next((p for p in self.ai_team if not p.is_fainted()), None)

    def active_player(self) -> Optional[Pokemon]:
        return next((p for p in self.player_team if not p.is_fainted()), None)

    def is_terminal(self) -> bool:
        return all(p.is_fainted() for p in self.ai_team) or all(p.is_fainted() for p in self.player_team)

    def copy(self) -> "BattleState":
        return deepcopy(self)


 
class BattleAI:
    """Battle AI that chooses actions using recursive simulation."""

    def __init__(self, recursion_depth: int = 2):
        self.stats_df, self.moves_df, self.abilities_df = load_local_data()
        self.recursion_depth = recursion_depth

    
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
        if "effect" in self.moves_df.columns:
            for _, row in self.moves_df.iterrows():
                eff = str(row.get("effect") or "").lower()
                name = row.get("move") or row.get("name")
                if not name:
                    continue
                if "rain" in eff or "heavy rain" in eff:
                    weather_moves[name] = "Rain"
                elif "sun" in eff or "harsh sunlight" in eff:
                    weather_moves[name] = "Sun"
                elif "sandstorm" in eff:
                    weather_moves[name] = "Sand"
                elif "snow" in eff or "hail" in eff:
                    weather_moves[name] = "Snow"
        else:
            # fallback mapping
            weather_moves = {
                "Rain Dance": "Rain",
                "Sunny Day": "Sun",
                "Sandstorm": "Sand",
                "Snowscape": "Snow",
            }
        return weather_moves

    
    def choose_ai_action(self, state: BattleState) -> Dict[str, Any]:
        """
        Return action dict:
        - {"type":"attack", "move": Move}
        - {"type":"setup", "move": Move, "weather": "Rain"/...}
        - {"type":"switch", "pokemon": Pokemon}
        - {"type":"noop"}
        """
        if state.is_terminal():
            return {"type": "noop"}

        ai_pkm = state.active_ai()
        pl_pkm = state.active_player()
        if not ai_pkm or not pl_pkm:
            return {"type": "noop"}

        setup_moves = self.get_setup_moves()
        weather_moves = self.get_weather_moves()

        
        rule = self._rule_layer_action(state, setup_moves, weather_moves)
        if rule is not None:
            return rule

        best_action = {"type": "attack", "move": ai_pkm.moves[0] if ai_pkm.moves else None}
        best_score = -float("inf")

        # Enumerate possible actions (attack moves + setup + possible switch candidates)
        possible_actions = []
        for mv in ai_pkm.moves:
            possible_actions.append({"type": "attack", "move": mv})
        for mv in ai_pkm.moves:
            if mv.name in weather_moves:
                possible_actions.append({"type": "setup", "move": mv, "weather": weather_moves[mv.name]})
            if mv.name in setup_moves:
                possible_actions.append({"type": "setup_stat", "move": mv})
        for candidate in state.ai_team:
            if candidate.name != ai_pkm.name and not candidate.is_fainted():
                possible_actions.append({"type": "switch", "pokemon": candidate})

        for action in possible_actions:
            sim_state = state.copy()
            next_state = self.simulate_action(
                sim_state,
                attacker=sim_state.active_ai(),
                defender=sim_state.active_player(),
                action=action,
                weather_moves=weather_moves,
            )
            score = self.recursive_evaluate(next_state, depth=self.recursion_depth - 1, maximizing=False)
            # penalize redundant weather setup (if same weather active with >1 turn remaining)
            if action["type"] == "setup" and state.weather.get("type") == action.get("weather") and state.weather.get("turns", 0) > 1:
                score -= 5.0
            if score > best_score:
                best_score = score
                best_action = action

        # Slight preference if item extends chosen weather and no weather is up yet
        if best_action.get("type") == "setup":
            if self._weather_item_synergy(ai_pkm, best_action.get("weather")) and not state.weather["type"]:
                best_action["reason"] = "item_synergy"

        return best_action

    
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

    
    def _tick_weather(self, state: BattleState, consumed_by_setup: bool) -> None:
        """Decrement weather turns by 1 per action unless just refreshed by setup."""
        if consumed_by_setup:
            return
        wt = state.weather.get("type")
        turns = state.weather.get("turns", 0)
        if wt and turns > 0:
            turns -= 1
            if turns <= 0:
                state.weather = {"type": None, "turns": 0}
            else:
                state.weather = {"type": wt, "turns": turns}

    
    def _stage_mult(self, stage: int) -> float:
        return (2 + stage) / 2.0 if stage >= 0 else 2.0 / (2 - stage)

    def _effective_speed(self, p: Pokemon) -> float:
        return float(p.speed) * self._stage_mult(p.stat_stages.get("speed", 0))

    def _outspeeds(self, a: Pokemon, d: Pokemon) -> bool:
        return self._effective_speed(a) >= self._effective_speed(d)

    def _best_opponent_move(self, defender: Pokemon, target: Pokemon, state: BattleState) -> Optional[Move]:
        if not defender or not defender.moves:
            return None
        scored = []
        for m in defender.moves:
            if m.pp <= 0:
                continue
            # Score by expected damage approximation
            dmg = self.calculate_damage(defender, target, m, state.weather)
            scored.append((dmg, m))
        return max(scored, key=lambda x: x[0])[1] if scored else None

    def _bench_candidates(self, state: BattleState, exclude_name: str) -> List[Pokemon]:
        return [p for p in state.ai_team if p.name != exclude_name and not p.is_fainted()]

    def _rule_layer_action(self, state: BattleState, setup_moves: List[str], weather_moves: Dict[str, str]) -> Optional[Dict[str, Any]]:
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
                dmg = self.calculate_damage(ai_pkm, pl_pkm, mv, state.weather)
                if dmg >= 0.6 * pl_pkm.hp and self._outspeeds(ai_pkm, pl_pkm):
                    # choose the strongest of these
                    if not best_lethal or dmg > best_lethal[0]:
                        best_lethal = (dmg, mv)
        if best_lethal is not None:
            return {"type": "attack", "move": best_lethal[1], "reason": "rule_super_effective_ko"}

        # 2) If opponent likely KOs us next turn, look for an immunity/resist switch; else best neutral tank
        opp_best = self._best_opponent_move(pl_pkm, ai_pkm, state)
        if opp_best is not None:
            opp_dmg = self.calculate_damage(pl_pkm, ai_pkm, opp_best, state.weather)
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
            opp_dmg = self.calculate_damage(pl_pkm, ai_pkm, opp_best, state.weather)
            if opp_dmg >= 0.5 * ai_pkm.hp and not self._outspeeds(ai_pkm, pl_pkm):
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
        if attacker.is_fainted():
            return state

        consumed_by_setup = False

        if action["type"] == "setup":
            mv = action.get("move")
            wt = action.get("weather")
            base_turns = 5
            # weather-extending items
            item = (getattr(attacker, "item", "") or "").strip()
            if (wt == "Sun" and item == "Heat Rock") or (wt == "Rain" and item == "Damp Rock") or (wt == "Sand" and item == "Smooth Rock") or (wt == "Snow" and item == "Icy Rock"):
                base_turns = 8
            state.weather = {"type": wt, "turns": base_turns}
            if isinstance(mv, Move):
                mv.pp = max(0, mv.pp - 1)
            consumed_by_setup = True
            self._tick_weather(state, consumed_by_setup=True)
            return state

        if action["type"] == "setup_stat":
            mv = action.get("move")
            name = (mv.name if isinstance(mv, Move) else str(mv)).lower()
            if "swords" in name:
                attacker.stat_stages["attack"] = min(6, attacker.stat_stages.get("attack", 0) + 2)
            elif "nasty plot" in name:
                attacker.stat_stages["special_attack"] = min(6, attacker.stat_stages.get("special_attack", 0) + 2)
            elif "calm mind" in name:
                attacker.stat_stages["special_attack"] = min(6, attacker.stat_stages.get("special_attack", 0) + 1)
                attacker.stat_stages["special_defense"] = min(6, attacker.stat_stages.get("special_defense", 0) + 1)
            elif "dragon dance" in name or "agility" in name:
                attacker.stat_stages["speed"] = min(6, attacker.stat_stages.get("speed", 0) + 1)
            elif "iron defense" in name or "acid armor" in name or "barrier" in name:
                attacker.stat_stages["defense"] = min(6, attacker.stat_stages.get("defense", 0) + 2)
            elif "amnesia" in name:
                attacker.stat_stages["special_defense"] = min(6, attacker.stat_stages.get("special_defense", 0) + 2)
            elif "cosmic power" in name:
                attacker.stat_stages["defense"] = min(6, attacker.stat_stages.get("defense", 0) + 1)
                attacker.stat_stages["special_defense"] = min(6, attacker.stat_stages.get("special_defense", 0) + 1)
            elif "bulk up" in name:
                attacker.stat_stages["attack"] = min(6, attacker.stat_stages.get("attack", 0) + 1)
                attacker.stat_stages["defense"] = min(6, attacker.stat_stages.get("defense", 0) + 1)
            if isinstance(mv, Move):
                mv.pp = max(0, mv.pp - 1)
            self._tick_weather(state, consumed_by_setup=False)
            return state

        if action["type"] == "switch":
            new_pkm = action.get("pokemon")
            team = state.ai_team if attacker in state.ai_team else state.player_team
            for i, p in enumerate(team):
                if p.name == new_pkm.name and not p.is_fainted():
                    team.insert(0, team.pop(i))
                    break
            self._tick_weather(state, consumed_by_setup=False)
            return state

        if action["type"] == "attack":
            mv = action.get("move")
            if mv is None:
                state.last_event = "But it failed!"
                self._tick_weather(state, consumed_by_setup=False)
                return state
            # accuracy check (accepts values > 1 as %)
            acc_val = (mv.accuracy or 100)
            acc_prob = acc_val / 100.0 if acc_val > 1 else acc_val
            if mv.pp <= 0:
                state.last_event = f"{attacker.name} tried to use {mv.name}, but it has no PP left!"
                self._tick_weather(state, consumed_by_setup=False)
                return state
            if random.random() > acc_prob:
                mv.pp = max(0, mv.pp - 1)
                state.last_event = f"{attacker.name} used {mv.name}, but it missed!"
                self._tick_weather(state, consumed_by_setup=False)
                return state
            # crit roll ~1/24
            is_crit = random.random() < (1.0 / 24.0)
            dmg = self.calculate_damage(attacker, defender, mv, state.weather)
            if is_crit:
                dmg = int(max(1, dmg) * 1.5)
            defender.hp = max(0, defender.hp - dmg)
            mv.pp = max(0, mv.pp - 1)
            # Life Orb recoil
            item = str(getattr(attacker, "item", "") or "")
            recoil_txt = ""
            if item.lower() == "life orb" and dmg > 0:
                recoil = max(1, int(attacker.max_hp * 0.1))
                attacker.hp = max(0, attacker.hp - recoil)
                recoil_txt = f" {attacker.name} is hurt by Life Orb! (-{recoil})"
            # Effectiveness text
            mult = get_type_multiplier(mv.type, defender.type)
            eff_txt = ""
            if mult == 0.0:
                eff_txt = " It had no effect."
            elif mult >= 2.0:
                eff_txt = " It's super effective!"
            elif mult <= 0.5:
                eff_txt = " It's not very effective."
            crit_txt = " Critical hit!" if is_crit else ""
            state.last_event = f"{attacker.name} used {mv.name}! It dealt {dmg} damage.{crit_txt}{eff_txt}{recoil_txt}"
            self._tick_weather(state, consumed_by_setup=False)
            return state

        # fallback noop
        self._tick_weather(state, consumed_by_setup=False)
        return state

    def calculate_damage(self, attacker: Pokemon, defender: Pokemon, move: Move, weather: Dict[str, Any]) -> int:
        """
        Simplified damage model:
         - uses physical stat (attack/defense) for physical moves
         - uses special attack/defense for special moves
         - applies STAB (1.5), type multiplier (from chart), weather multipliers, and simple random variance
         - honors Air Balloon/Levitate immunity to Ground and Utility Umbrella to weather
        """
        if move.power <= 0:
            return 0

        # Ground immunities via item/ability
        if _norm_type(move.type) == "Ground":
            if getattr(defender, "item", "") == "Air Balloon" or getattr(defender, "ability", "") == "Levitate":
                return 0

        utility_umbrella = (getattr(attacker, "item", "") == "Utility Umbrella")

        # base stats
        is_special = bool(getattr(move, "is_special", False) or str(getattr(move, "category", "")).lower() == "special")
        atk_stat = attacker.special_attack if is_special else attacker.attack
        def_stat = defender.special_defense if is_special else defender.defense

        # apply stat stages (simplified)
        def stage_mult(stage: int) -> float:
            return (2 + stage) / 2.0 if stage >= 0 else 2.0 / (2 - stage)

        atk_stat *= stage_mult(attacker.stat_stages.get("special_attack" if is_special else "attack", 0))
        def_stat *= stage_mult(defender.stat_stages.get("special_defense" if is_special else "defense", 0))

        # STAB (Adaptability makes it 2.0)
        adaptability = str(getattr(attacker, "ability", "") or "").lower() == "adaptability"
        stab = 2.0 if adaptability and any(_norm_type(move.type) == _norm_type(t) for t in attacker.type) else (
            1.5 if any(_norm_type(move.type) == _norm_type(t) for t in attacker.type) else 1.0
        )

        # type multiplier (dual-type aware)
        type_mult = get_type_multiplier(move.type, defender.type)

        # weather multiplier
        weather_mult = 1.0
        if weather and weather.get("type") and not utility_umbrella:
            wt = weather["type"]
            if wt == "Rain":
                if _norm_type(move.type) == "Water":
                    weather_mult = 1.5
                elif _norm_type(move.type) == "Fire":
                    weather_mult = 0.5
            elif wt == "Sun":
                if _norm_type(move.type) == "Fire":
                    weather_mult = 1.5
                elif _norm_type(move.type) == "Water":
                    weather_mult = 0.5
            elif wt == "Sand":
                if _norm_type(move.type) == "Rock":
                    weather_mult = 1.3
            elif wt == "Snow":
                if _norm_type(move.type) == "Ice":
                    weather_mult = 1.3

        # Item/ability modifiers
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

        # Sheer Force (if move has an effect field)
        if str(getattr(attacker, "ability", "") or "").lower() == "sheer force" and getattr(move, "effect", None):
            item_mult *= 1.3

        # simple damage formula with random variance
        level_factor = (2 * attacker.level) / 5 + 2
        base = ((level_factor * move.power * (atk_stat / max(1.0, def_stat))) / 50.0) + 2.0
        variance = random.uniform(0.85, 1.0)
        dmg = base * stab * type_mult * weather_mult * item_mult * variance
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
