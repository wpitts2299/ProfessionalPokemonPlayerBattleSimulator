"""
Entry point.

Default: launches the Tkinter GUI battle (gui_battle.py).
Optional: run with --cli to use the console interactive battle.
"""
import argparse
from typing import List
from data_loader import load_local_data, Move, Pokemon, create_pokemon_from_name
from team_builder import generate_balanced_team, pick_item
from pikalytics_util import fetch_overview
from battle_ai import BattleAI, BattleState

def print_team(team, title="Team"):
    print(f"=== {title} ===")
    for p in team:
        print(f"- {p.name} ({'/'.join(p.type)}) | Item: {p.item} | Ability: {p.ability}")
        for mv in p.moves:
            print(f"   - {mv.name} [{mv.type}] pow={mv.power} acc={mv.accuracy} pp={mv.pp}")

def print_status(state: BattleState):
    ai = state.active_ai()
    pl = state.active_player()
    wt = state.weather.get("type") or "None"
    turns = state.weather.get("turns", 0)
    print("\n=== Battle Status ===")
    if ai:
        print(f"AI:     {ai.name} HP {ai.hp}/{ai.max_hp} | Types: {'/'.join(ai.type)} | Item: {ai.item}")
    if pl:
        print(f"Player: {pl.name} HP {pl.hp}/{pl.max_hp} | Types: {'/'.join(pl.type)} | Item: {pl.item}")
    print(f"Weather: {wt} (turns: {turns})\n")


def choose_player_action(state: BattleState, ai: BattleAI):
    player = state.active_player()
    if not player:
        return {"type": "noop"}

    # List moves
    print("Your moves:")
    for idx, mv in enumerate(player.moves, 1):
        print(f"  {idx}. {mv.name} [{mv.type}] pow={mv.power} acc={mv.accuracy} pp={mv.pp}")
    # List switch options
    bench: List[Pokemon] = [p for p in state.player_team if p.name != player.name and not p.is_fainted()]
    if bench:
        print("Switch options:")
        for idx, p in enumerate(bench, 1):
            print(f"  s{idx}. Switch to {p.name} ({'/'.join(p.type)}) HP {p.hp}/{p.max_hp}")

    cmd = input("Choose action (1-4 move, s# switch): ").strip().lower()
    weather_moves = ai.get_weather_moves()
    setup_moves = ai.get_setup_moves()

    # Switch handling
    if cmd.startswith("s") and len(cmd) > 1 and bench:
        try:
            bi = int(cmd[1:]) - 1
            if 0 <= bi < len(bench):
                return {"type": "switch", "pokemon": bench[bi]}
        except Exception:
            pass

    # Move handling
    try:
        mi = int(cmd) - 1
    except Exception:
        mi = 0
    mi = max(0, min(mi, len(player.moves) - 1))
    mv = player.moves[mi]
    if mv.name in weather_moves:
        return {"type": "setup", "move": mv, "weather": weather_moves[mv.name]}
    if mv.name in setup_moves:
        return {"type": "setup_stat", "move": mv}
    return {"type": "attack", "move": mv}


def run_cli_battle():
    stats_df, moves_df, abilities_df = load_local_data()
    print("Loaded data.")

    # Restrict AI pool to >=0.5% usage in Pikalytics (format gen9vgc2025regh)
    # Prefer CSV compendium if available; else try HTML cache/fetch.
    allowed = set()
    try:
        from pikalytics_util import load_compendium_single_csv, load_compendium_csv, CACHE_DIR
        import os
        single_csv = os.path.join(CACHE_DIR, "compendium_gen9vgc2025regh.csv")
        if os.path.exists(single_csv):
            comp = load_compendium_single_csv("gen9vgc2025regh")
            allowed = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 0.5}
        elif os.path.exists(os.path.join(CACHE_DIR, "compendium_gen9vgc2025regh_overview.csv")):
            comp = load_compendium_csv("gen9vgc2025regh")
            allowed = {name for name, data in comp.get("pokemon", {}).items() if data.get("usage", 0) >= 0.5}
        else:
            overview = fetch_overview("gen9vgc2025regh")
            allowed = {name for name, usage in overview if usage >= 0.5}
    except Exception:
        allowed = set()
    ai_team = generate_balanced_team(
        stats_df, moves_df, abilities_df, n=6, item_style="aggressive",
        allowed_names=(allowed if allowed else None)
    )
    player_team = None
    # Parse optional player team from args (comma-separated)
    import sys
    names_arg = None
    style = "balanced"
    for i, tok in enumerate(sys.argv):
        if tok.startswith("--player-team="):
            names_arg = tok.split("=",1)[1]
        elif tok.startswith("--item-style="):
            style = tok.split("=",1)[1]
    if names_arg:
        chosen = [x.strip() for x in names_arg.split(',') if x.strip()]
        team = []
        for nm in chosen[:6]:
            try:
                p = create_pokemon_from_name(nm, stats_df, moves_df, abilities_df, preferred_item=pick_item(style))
                team.append(p)
            except Exception:
                pass
        if team:
            player_team = team
    if player_team is None:
        player_team = generate_balanced_team(stats_df, moves_df, abilities_df, n=6, item_style="balanced")

    print_team(ai_team, "AI Team (balanced)")
    print()
    print_team(player_team, "Player Team (balanced)")

    state = BattleState(ai_team, player_team)
    ai = BattleAI(recursion_depth=2)

    print("\nStarting battle! You control the Player side.\n")
    turn = 1
    max_turns = 50
    while turn <= max_turns and not state.is_terminal():
        print(f"\n===== Turn {turn} =====")
        print_status(state)
        # Player acts
        p_action = choose_player_action(state, ai)
        state = ai.simulate_action(state, attacker=state.active_player(), defender=state.active_ai(), action=p_action, weather_moves=ai.get_weather_moves())
        if getattr(state, 'last_event', None):
            print(state.last_event)
        if state.is_terminal():
            break
        # AI acts
        a_action = ai.choose_ai_action(state)
        print(f"AI chose: {a_action}")
        state = ai.simulate_action(state, attacker=state.active_ai(), defender=state.active_player(), action=a_action, weather_moves=ai.get_weather_moves())
        if getattr(state, 'last_event', None):
            print(state.last_event)
        turn += 1

    print_status(state)
    ai_alive = any(not p.is_fainted() for p in state.ai_team)
    pl_alive = any(not p.is_fainted() for p in state.player_team)
    if pl_alive and not ai_alive:
        print("You win!")
    elif ai_alive and not pl_alive:
        print("AI wins!")
    else:
        print("Battle ended (turn limit or stalemate).")


def run_gui_battle():
    try:
        from gui_battle import BattleGUI
    except Exception as e:
        print("GUI unavailable (", e, ") — falling back to CLI.")
        return run_cli_battle()
    app = BattleGUI()
    app.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon battle runner")
    parser.add_argument("--cli", action="store_true", help="Run console interactive battle instead of GUI")
    parser.add_argument("--gui", action="store_true", help="Force GUI (default)")
    parser.add_argument("--player-team", help="Comma-separated list of 1-6 Pokemon names for the player team", default=None)
    parser.add_argument("--item-style", help="Item style for player team (balanced/aggressive/defensive)", default="balanced")
    args = parser.parse_args()

    if args.cli:
        run_cli_battle()
    else:
        run_gui_battle()
