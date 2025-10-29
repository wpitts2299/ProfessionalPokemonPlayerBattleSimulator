# main.py
"""
Main entry point for Pokémon AI Battle Simulator.

- Loads Pokémon, moves, and abilities data.
- Builds balanced AI and player teams automatically.
- Initializes the BattleState.
- Runs the AI’s move selection and displays results.
"""

from data_loader import load_local_data
from team_builder import generate_balanced_team
from battle_ai import BattleAI, BattleState


if __name__ == "__main__":
    print("=== Loading Data ===")
    stats_df, moves_df, abilities_df = load_local_data()
    print(f"Loaded {len(stats_df)} Pokémon, {len(moves_df)} moves, and {len(abilities_df)} abilities.\n")

    print("=== Building Teams ===")
    ai_team = generate_balanced_team(stats_df, moves_df, abilities_df)
    player_team = generate_balanced_team(stats_df, moves_df, abilities_df)
    print("AI Team:")
    for p in ai_team:
        print(f" - {p.name} ({'/'.join(p.type)}) | Moves: {[m.name for m in p.moves]}")
    print("\nPlayer Team:")
    for p in player_team:
        print(f" - {p.name} ({'/'.join(p.type)}) | Moves: {[m.name for m in p.moves]}")

    print("\n=== Initializing Battle ===")
    state = BattleState(ai_team, player_team)
    ai = BattleAI()

    print("=== AI Choosing Move ===")
    action = ai.choose_ai_action(state)
    if action["type"] == "attack":
        print(f"AI chose to attack using {action['move'].name}")
    elif action["type"] == "setup":
        print(f"AI chose to set up weather: {action['weather']} with {action['move'].name}")
    else:
        print("AI chose to wait (noop).")

    print("\n=== Battle Simulation Complete ===")
