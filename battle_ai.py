# battle_ai.py

"""""
-------------

Framework for a weather- and ability-aware Pokémon battle AI.

This module defines the BattleAI class, which:
- Reads data from data_loader.py
- Simulates simplified battle states
- Uses recursive evaluation to choose actions
- Accounts for weather, abilities, and items
"""

from copy import deepcopy
import random

# You’ll import your data structures
from data_loader import Pokemon, Move
# (Optional: from team_builder import Team)


class BattleState:
    """Simplified representation of a turn’s state."""
    def __init__(self, ai_team, player_team, weather=None):
        self.ai_team = ai_team
        self.player_team = player_team
        self.ai_active = ai_team[0]
        self.player_active = player_team[0]
        self.weather = weather or {"type": None, "turns": 0}


class BattleAI:
    """Core decision engine for weather- and ability-aware battle simulation."""

    def __init__(self, recursion_depth=2):
        self.recursion_depth = recursion_depth

    # ------------------------------------------------------------------
    # === Decision Logic ===
    # ------------------------------------------------------------------

    def choose_ai_action(self, battle_state):
        """Main entry point. Chooses best action for the AI."""
        possible_actions = self.get_possible_actions(battle_state, ai_side=True)
        best_action, best_score = None, float("-inf")

        for action in possible_actions:
            # Project future state
            projected = self.simulate_action(battle_state, action, ai_side=True)
            score = self.recursive_evaluate(projected, depth=self.recursion_depth)

            # Add any bonuses (synergy, setup preference, etc.)
            if "priority_bonus" in action:
                score += action["priority_bonus"]

            if score > best_score:
                best_action, best_score = action, score

        return best_action

    # ------------------------------------------------------------------
    # === Action Enumeration ===
    # ------------------------------------------------------------------

    def get_possible_actions(self, state, ai_side=True):
        """
        Returns all legal actions:
        - Offensive moves
        - Status/setup moves (e.g. weather)
        - Switches (if advantageous)
        """
        actor = state.ai_active if ai_side else state.player_active
        actions = []

        # Offensive moves
        for move in actor.moves:
            if move.category in ["physical", "special"]:
                actions.append({"type": "attack", "move": move})

        # Setup moves (weather/status)
        for move in actor.moves:
            if move.category == "status" and move.name in [
                "Rain Dance", "Sunny Day", "Sandstorm", "Snowscape"
            ]:
                actions.append({"type": "setup", "weather": move.name.split()[0]})

        # Example switch logic placeholder
        for p in state.ai_team:
            if p != actor and not p.is_fainted:
                actions.append({"type": "switch", "target": p})

        return actions

    # ------------------------------------------------------------------
    # === Recursive Simulation ===
    # ------------------------------------------------------------------

    def recursive_evaluate(self, state, depth):
        """
        Predicts future states recursively and assigns a numeric score.
        Higher = better for AI.
        """
        if depth == 0:
            return self.evaluate_state(state)

        best_outcome = float("-inf")
        for action in self.get_possible_actions(state, ai_side=True):
            next_state = self.simulate_action(state, action, ai_side=True)
            outcome = self.recursive_evaluate(next_state, depth - 1)
            best_outcome = max(best_outcome, outcome)

        return best_outcome

    # ------------------------------------------------------------------
    # === State Evaluation ===
    # ------------------------------------------------------------------

    def evaluate_state(self, state):
        """
        Assigns a numerical score to a battle state.
        Takes into account HP, fainted Pokémon, weather synergy, etc.
        """
        score = 0
        ai_alive = [p for p in state.ai_team if not p.is_fainted]
        pl_alive = [p for p in state.player_team if not p.is_fainted]

        # Simple HP evaluation
        score += sum(p.hp for p in ai_alive)
        score -= sum(p.hp for p in pl_alive)

        # Placeholder: weather/ability synergy bonuses
        if state.weather and state.weather.get("type"):
            w = state.weather["type"]
            # Example: reward Swift Swim in Rain
            for p in ai_alive:
                if p.ability == "Swift Swim" and w == "Rain":
                    score += 40
            for p in pl_alive:
                if p.ability == "Swift Swim" and w == "Rain":
                    score -= 40

        return score

    # ------------------------------------------------------------------
    # === Action Simulation ===
    # ------------------------------------------------------------------

    def simulate_action(self, state, action, ai_side):
        """Returns a projected copy of the battle state after one action."""
        new_state = deepcopy(state)
        actor = new_state.ai_active if ai_side else new_state.player_active
        target = new_state.player_active if ai_side else new_state.ai_active

        # Simplified mechanics:
        if action["type"] == "attack":
            move = action["move"]
            dmg = self.simulate_damage(actor, target, move, new_state)
            target.hp = max(0, target.hp - dmg)
            if target.hp == 0:
                target.is_fainted = True

        elif action["type"] == "setup":
            new_state.weather = {"type": action["weather"], "turns": 5}

        elif action["type"] == "switch":
            if ai_side:
                new_state.ai_active = action["target"]
            else:
                new_state.player_active = action["target"]

        return new_state

    # ------------------------------------------------------------------
    # === Damage Simulation (simplified) ===
    # ------------------------------------------------------------------

    def simulate_damage(self, user, target, move, state):
        """Placeholder damage model with weather and ability modifiers."""
        atk = user.attack if move.category == "physical" else user.spattack
        defense = target.defense if move.category == "physical" else target.spdef

        # Example: basic weather modifier
        weather_mult = 1.0
        if state.weather:
            w = state.weather.get("type")
            if w == "Rain" and move.type == "Water":
                weather_mult = 1.5
            elif w == "Rain" and move.type == "Fire":
                weather_mult = 0.5
            elif w == "Sun" and move.type == "Fire":
                weather_mult = 1.5
            elif w == "Sun" and move.type == "Water":
                weather_mult = 0.5

        # Placeholder type multiplier
        type_mult = 1.0  # integrate from your type chart

        base_damage = ((2 * user.level / 5 + 2) * move.power * (atk / defense)) / 50 + 2
        damage = int(base_damage * weather_mult * type_mult)
        return max(1, damage)
